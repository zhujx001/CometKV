"""Exact selector regressions, including adversarial radix boundaries and graph replay."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CometKV top-k tests require CUDA", allow_module_level=True)

from cometkv import exact_topk_indices_into


def buffers(scores, k):
    rows, tokens = scores.shape
    return (
        torch.full((rows, k + 13), -7, device=scores.device, dtype=torch.int32),
        torch.empty((rows, (tokens + 1023) // 1024, 288), device=scores.device, dtype=torch.int32),
        torch.empty_like(scores, dtype=torch.int32),
        torch.empty((rows, 7), device=scores.device, dtype=torch.int32),
    )


def check_selection(scores, indices, k, start, end):
    # Stable descending sort independently specifies ties: earlier token ID wins.
    expected = scores[:, start:end].argsort(dim=1, descending=True, stable=True)[:, :k] + start
    assert torch.equal(indices[:, :k].long(), expected.sort(1).values)
    assert (indices[:, k:] == -7).all()


@pytest.mark.parametrize("tokens,k,start,trim,kind", [
    (257, 0, 4, 3, "random"), (257, 1, 4, 3, "random"),
    (257, 250, 4, 3, "random"), (8192, 163, 4, 32, "random"),
    (32769, 655, 7, 37, "random"), (65539, 1310, 4, 32, "random"),
    (8193, 4096, 2, 1, "random"), (8193, 655, 4, 32, "equal"),
    (8193, 655, 4, 32, "adjacent"), (8193, 655, 4, 32, "ties"),
    (32769, 655, 4, 32, "narrow"),
    (8193, 655, 4, 32, "special"),
])
def test_exact_topk_matches_stable_sort(tokens, k, start, trim, kind):
    torch.manual_seed(93)
    scores = torch.randn((8, tokens), device="cuda", dtype=torch.float32)
    if kind == "equal":
        scores.fill_(-3.0)
    elif kind == "adjacent":
        scores.fill_(1.0)
        scores.view(torch.int32).add_(torch.randint(64, scores.shape, device="cuda", dtype=torch.int32))
    elif kind == "ties":
        scores.round_()
    elif kind == "narrow":
        scores.mul_(1e-3).add_(-8.)
    elif kind == "special":
        scores[:, ::5] = float("nan")
        scores[:, 1::5] = float("inf")
        scores[:, 2::5] = -float("inf")
        scores[1].fill_(-float("inf"))
        scores[2].fill_(float("nan"))
        scores[3].zero_()
        scores[3, ::2] = -0.0
    end = tokens - trim
    scores[:, :start] = float("inf")
    scores[:, end:] = float("inf")
    indices, hist, work, state = buffers(scores, k)
    exact_topk_indices_into(scores, indices, hist, work, state, k, start, end)
    check_selection(scores, indices, k, start, end)


def test_topk_cuda_graph_replays_changed_scores():
    scores = torch.randn((8, 32768), device="cuda", dtype=torch.float32)
    k, start, end = 655, 4, 32736
    indices, hist, work, state = buffers(scores, k)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        exact_topk_indices_into(scores, indices, hist, work, state, k, start, end)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        exact_topk_indices_into(scores, indices, hist, work, state, k, start, end)
    for seed in (17, 0, 542):
        torch.manual_seed(seed)
        scores.normal_()
        if seed == 0:
            scores[::2].zero_()  # Mixed flat/non-flat rows, then back to ordinary scores.
        graph.replay()
        check_selection(scores, indices, k, start, end)


def test_topk_rejects_excess_budget():
    scores = torch.zeros((1, 8192), device="cuda")
    args = buffers(scores, 4097)
    with pytest.raises(RuntimeError, match="invalid k"):
        exact_topk_indices_into(scores, *args, 4097, 0, 8192)
