import importlib.util
import json
from pathlib import Path



def load_eval_module(monkeypatch):
    module_path = Path(__file__).resolve().with_name('eval.py')
    monkeypatch.chdir(module_path.parent)
    spec = importlib.util.spec_from_file_location('longbench_eval', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def test_collect_scores_can_filter_single_task(monkeypatch, tmp_path):
    eval_mod = load_eval_module(monkeypatch)
    result_dir = tmp_path / 'results' / 'pred' / 'llama-3.1-8b' / 'CometKV'
    result_dir.mkdir(parents=True)
    qasper = {
        'pred': 'yes',
        'answers': ['yes'],
        'all_classes': None,
        'length': 100,
    }
    hotpot = {
        'pred': 'no',
        'answers': ['yes'],
        'all_classes': None,
        'length': 100,
    }
    (result_dir / 'qasper.jsonl').write_text(json.dumps(qasper) + '\n', encoding='utf-8')
    (result_dir / 'hotpotqa.jsonl').write_text(json.dumps(hotpot) + '\n', encoding='utf-8')

    scores = eval_mod.collect_scores(result_dir, use_longbench_e=False, task='qasper')

    assert list(scores.keys()) == ['qasper']
    assert scores['qasper'] == 100.0



def test_merge_result_file_updates_single_task_score(tmp_path, monkeypatch):
    eval_mod = load_eval_module(monkeypatch)
    result_path = tmp_path / 'result.json'
    result_path.write_text(
        json.dumps({'qasper': 12.34, 'hotpotqa': 56.78}, ensure_ascii=False, indent=4),
        encoding='utf-8',
    )

    merged = eval_mod.merge_result_file(result_path, {'qasper': 98.76})

    assert merged == {'qasper': 98.76, 'hotpotqa': 56.78}
    assert json.loads(result_path.read_text(encoding='utf-8')) == merged


def test_parse_args_accepts_mistral(monkeypatch):
    eval_mod = load_eval_module(monkeypatch)

    args = eval_mod.parse_args([
        '--model', 'mistral-7b-instruct-v0.2',
        '--attn_type', 'CometKV',
        '--task', 'qasper',
    ])

    assert args.model == 'mistral-7b-instruct-v0.2'
