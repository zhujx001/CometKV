import os
import time
import torch
import flashinfer
from termcolor import colored

PREFILL_MLP_CHUNK_SIZE = 32768


def normalize_eos_token_ids(eos_token_ids):
    if eos_token_ids is None:
        return []
    if isinstance(eos_token_ids, (list, tuple, set)):
        normalized_ids = [int(token_id) for token_id in eos_token_ids]
    else:
        normalized_ids = [int(eos_token_ids)]

    deduped_ids = []
    for token_id in normalized_ids:
        if token_id not in deduped_ids:
            deduped_ids.append(token_id)
    return deduped_ids


class LLM:
    """
    A class representing the LLM (currently support Llama and Qwen).
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str
    ) -> None:
        """ Initializes the LLM.
        Args:
            model_name (str): The name of the model.
            max_length (int): The maximum length (prefill+decode) of sequences.
            dtype (torch.dtype): The data type for model computations.
            device_map (str): The device for model, suppor 'cuda:x' or 'auto (automatically use all visible GPUs)'.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map
        self.event_profile_enabled = os.environ.get("COMETKV_EVENT_PROFILE", "0").lower() in ("1", "true", "yes")
        self.event_profile = {}

    def _event_profile_start(self):
        if not self.event_profile_enabled:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _event_profile_end(self, name, start):
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.event_profile.setdefault(name, []).append((start, end))

    def event_profile_summary_s(self):
        return {
            name: sum(start.elapsed_time(end) for start, end in pairs) / 1000.0
            for name, pairs in self.event_profile.items()
        }


    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        # print(f'Layer = {layer_idx}, start_bdx = {start_bdx}')

        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]
        
        temp_hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        del temp_hidden_states
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim) # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        key_states, value_states = self.kv_cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx, start_bdx)
        temp_attn_out = self.prefill_attention(query_states, key_states, value_states, layer_idx)
        self.kv_cache.sync(layer_idx, start_bdx)
        del query_states, key_states, value_states

        hidden_states += self.wo(temp_attn_out, layer, bsz, seq_len, dim)
        del temp_attn_out

        # post attention
        residual = hidden_states.clone()

        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        # faster when split batches
        for batch_idx in range(0, bsz, 1):
            # chunk for lower memory comsumption, especially for 1M context
            for start_idx in range(0, seq_len, PREFILL_MLP_CHUNK_SIZE):
                end_idx = min(seq_len, start_idx + PREFILL_MLP_CHUNK_SIZE)
                hidden_states[batch_idx:batch_idx+1, start_idx:end_idx, :] = self.mlp(hidden_states[batch_idx:batch_idx+1, start_idx:end_idx, :], layer)

        hidden_states += residual
        del residual

        return hidden_states


    def layer_decode(self, layer_idx, hidden_states):
        # print(f'Layer = {layer_idx}')

        _layer_event_t = self._event_profile_start()
        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        # assert seq_len == 1, f"Error: seq_len should be 1 for decoding, but got {seq_len}."
        layer = self.layers[layer_idx]

        _event_t = self._event_profile_start()
        hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        self._event_profile_end("decode_input_layernorm", _event_t)

        _event_t = self._event_profile_start()
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        self._event_profile_end("decode_qkv_rope", _event_t)

        _event_t = self._event_profile_start()
        key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
        self._event_profile_end("decode_update_kv_cache", _event_t)

        _event_t = self._event_profile_start()
        attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
        self._event_profile_end("decode_attention", _event_t)

        _event_t = self._event_profile_start()
        hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
        hidden_states = residual + hidden_states
        self._event_profile_end("decode_o_proj_residual", _event_t)

        _event_t = self._event_profile_start()
        residual = hidden_states
        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states
        self._event_profile_end("decode_mlp_block", _event_t)
        self._event_profile_end("decode_layer_total", _layer_event_t)

        return hidden_states


    def prefill_forward(self, inputs_ids):
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device).contiguous()
        for start_bdx in range(0, bsz, self.prefill_bsz):
            end_bdx = min(bsz, start_bdx + self.prefill_bsz)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])  # [prefill_batch_size, seq_len, hidden_size]

            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    hidden_states = self.parameter_move(hidden_states, ldx)
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(self.layers[0].device)
            else:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
        
        last_hidden_states = self.layernorm(last_hidden_states, self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden_states)
        
        return logits
        

    def decode_forward(self, inputs_ids):
        _event_t = self._event_profile_start()
        hidden_states = self.word_embedding(inputs_ids)
        self._event_profile_end("decode_embedding", _event_t)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)

        _event_t = self._event_profile_start()
        hidden_states = self.layernorm(hidden_states, self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        self._event_profile_end("decode_final_norm_lm", _event_t)
        
        return logits


    # ---------------- CUDA-graph decode (CometKV, single-GPU, lockstep same-length) ----------------
    def _cg_supported(self, use_cuda_graph):
        from cache_hub import cometkv_cache
        return bool(
            use_cuda_graph
            and self.attention_type in ("CometKV", "CometKV_GPU")
            and self.num_gpus == 1
            and isinstance(self.kv_cache, cometkv_cache)
            and not self.event_profile_enabled
            and not getattr(self.kv_cache, "profile_decode_update", False)
        )

    def _cg_setup(self):
        # Static IO + device position buffers (fixed addresses the captured graph reads/writes), and
        # pre-allocate the cache's fixed-width concat buffers so nothing allocates inside the capture.
        device = self.layers[0].device
        self._cg_input_ids = torch.empty((self.batch_size, 1), dtype=torch.int64, device=device)
        self._cg_position_ids = torch.zeros((self.batch_size,), dtype=torch.int32, device=device)
        for ldx in range(self.num_layers):
            self.kv_cache._ensure_cg_concat_buffers(ldx)
        self._cg_graph = None
        self._cg_regime = None
        self._cg_logits = None
        self._cg_pool = torch.cuda.graph_pool_handle()
        self._cg_warmup_steps = 2      # initial eager(normal) steps
        self._cg_warm_needed = 3       # cg-path eager steps before the first capture
        self._cg_warm_done = 0

    def _set_cg_mode(self, flag):
        self.use_cuda_graph = flag            # read by position_embedd (device RoPE position)
        self.kv_cache.use_cuda_graph = flag   # read by decode_update_kv_cache / sparse_attention

    def _cg_regime_key(self):
        # Includes the committed retrieval-range bounds: the selector bakes them into the
        # captured graph (host ints), so a plan change without a static/topk change must still recapture.
        return (int(self.kv_cache.fixed_prompt_local_static_length_host),
                int(self.kv_cache.active_sparse_len_host),
                int(getattr(self.kv_cache, "plan_range1_start_host", 0)),
                int(getattr(self.kv_cache, "plan_range1_end_host", 0)))

    def _cg_set_inputs(self, output_ids):
        self._cg_input_ids.copy_(output_ids.to(torch.int64).view(self.batch_size, 1))
        self._cg_position_ids.fill_(int(self.kv_cache.context))  # absolute position of this step's token
        self.kv_cache.cg_set_step_inputs()

    def _cg_replay(self, output_ids):
        self._set_cg_mode(True)
        self._cg_set_inputs(output_ids)
        self._cg_graph.replay()
        self.kv_cache.cg_advance_host_step()
        return self._cg_logits

    def _cg_capture(self, output_ids):
        self._set_cg_mode(True)
        self._cg_set_inputs(output_ids)
        # Documented torch.cuda.graph prerequisite: warm up the EXACT fn on a side stream right before
        # capture so cuBLAS/flashinfer/allocator workspaces bind to the graph's pool. Warming with the
        # SAME fixed inputs and WITHOUT advancing host counters is idempotent here (the append re-writes
        # the same decode-hot slot with the same key; the gather re-populates the same token-cache slots),
        # so no state rollback is needed.
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                self.decode_forward(inputs_ids=self._cg_input_ids)
        torch.cuda.current_stream().wait_stream(warm)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._cg_logits = self.decode_forward(inputs_ids=self._cg_input_ids)
        # Capture records the work but does not leave valid results in the output tensor; replay once
        # (idempotent: same inputs/decode_step => same append slot + same gather) to materialize this
        # step's logits before the loop samples from them.
        g.replay()
        self.kv_cache.cg_advance_host_step()
        self._cg_graph = g
        self._cg_regime = self._cg_regime_key()
        return self._cg_logits

    def _cg_warmup_step(self, output_ids):
        # cg path run EAGERLY (not captured): materializes flash cache_seqlens workspace + concat path so
        # the subsequent capture allocates nothing. Counts as a real decode step.
        self._set_cg_mode(True)
        self._cg_set_inputs(output_ids)
        logits = self.decode_forward(inputs_ids=self._cg_input_ids)
        self.kv_cache.cg_advance_host_step()
        return logits

    def _decode_step(self, output_ids, step_idx):
        if not getattr(self, "_cg_enabled", False):
            return self.decode_forward(inputs_ids=output_ids)
        if os.environ.get("COMETKV_CG_EAGER", "0") == "1":
            return self._cg_warmup_step(output_ids)  # DEBUG: cg path run eagerly, no capture/replay
        cache = self.kv_cache
        slide = cache.cg_is_slide_step()
        if (not slide) and self._cg_graph is not None and self._cg_regime == self._cg_regime_key():
            return self._cg_replay(output_ids)
        if (not slide) and step_idx >= self._cg_warmup_steps:
            # Run several cg-path-EAGER steps first to stabilize cublas/flash workspaces + allocator,
            # then capture (real steps, consuming decode budget). Insufficient warmup -> graph-pool
            # aliasing -> garbage on replay.
            if self._cg_warm_done < self._cg_warm_needed:
                self._cg_warm_done += 1
                return self._cg_warmup_step(output_ids)
            return self._cg_capture(output_ids)
        # eager (initial warmup steps, or window-slide steps which change the regime/plan)
        self._set_cg_mode(False)
        logits = self.decode_forward(inputs_ids=output_ids)
        if slide:
            self._cg_graph = None       # regime changed; re-warm + re-capture on the next steady steps
            self._cg_warm_done = self._cg_warm_needed  # already warm; capture directly next steady step
        return logits


    def sampling(self, logits, do_sample=False, temperature=0.6, top_p=0.95, top_k=20):
        if not do_sample:
            output_ids = logits.argmax(dim=-1)  # [bsz, 1], torch.int64
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)  # [bsz, 1, vocab_size]
            probs = probs.squeeze(1) # [bsz, vocab_size]
            if top_k != 0:
                output_ids = flashinfer.sampling.top_k_top_p_sampling_from_probs(probs, top_p=top_p, top_k=top_k)
            else:
                output_ids = flashinfer.sampling.top_p_sampling_from_probs(probs, top_p=top_p)
            output_ids = output_ids.unsqueeze(1) # [bsz, 1], torch.int32

        return output_ids


    def inference(self, inputs_ids, do_sample=False, temperature=0.6, top_p=0.95, top_k=20,
                  ignore_eos=True, eos_token_ids=None, collect_step_latency=False, profile_timing=False,
                  ignore_first_steps_for_tpot=0, use_cuda_graph=False):
        self.decode_step_latencies_ms = []
        self.generate_latency_stats = {}
        
        # Prefilling
        if profile_timing:
            print("Start prefilling ...")
            torch.cuda.synchronize()
            end2end_start = prefill_start = time.time()

        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = self.sampling(logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
        generated_ids = torch.empty(
            (self.batch_size, self.max_new_length),
            dtype=output_ids.dtype,
            device=output_ids.device,
        )
        generated_ids[:, 0:1] = output_ids
        generated_count = 1
        if profile_timing:
            torch.cuda.synchronize()
            first_token_end = time.time()
        self.move()

        if profile_timing:
            torch.cuda.synchronize()
            prefill_end = time.time()
            ttft_duration = first_token_end - prefill_start
            prefill_duration = prefill_end - prefill_start
            self.generate_latency_stats.update({
                "ttft_s": ttft_duration,
                "prefill_s": prefill_duration,
            })
            print(colored(f"TTFT latency: {round(ttft_duration, 4)} s", 'green'))
            print(colored(f"Prefilling latency: {round(prefill_duration, 4)} s", 'green'))

        # check if get EOS token during decoding
        if not ignore_eos:
            end_of_text = torch.zeros((self.batch_size, 1), dtype=torch.bool, device=inputs_ids.device)
            token_id_dtype = torch.int64 if not do_sample else torch.int32  # flashinfer returns int32
            eos_token_ids = normalize_eos_token_ids(eos_token_ids if eos_token_ids is not None else self.tokenizer.eos_token_id)
            if not eos_token_ids:
                raise ValueError("EOS stopping is enabled, but no eos_token_ids were provided.")
            eos_tokens = torch.tensor(eos_token_ids, dtype=token_id_dtype, device=inputs_ids.device)
            end_of_text |= torch.isin(output_ids, eos_tokens)
        
        # Decoding
        if profile_timing:
            print("Start decoding ...")
            ignored_decode_steps_target = max(int(ignore_first_steps_for_tpot), 0)
            ignored_decode_start = None
            ignored_decode_end = None
            decode_start = time.time()

        self._cg_enabled = self._cg_supported(use_cuda_graph)
        if self._cg_enabled:
            self._cg_warmed = False
            self._set_cg_mode(False)
            self._cg_setup()

        if not ignore_eos and end_of_text.all():
            print(colored("All sequences have reached EOS token, stop decoding.", 'yellow'))
        else:
            for decode_step_idx in range(self.max_new_length-1):
                if profile_timing and decode_step_idx == 0 and ignored_decode_steps_target > 0:
                    torch.cuda.synchronize()
                    ignored_decode_start = time.time()
                if profile_timing and decode_step_idx == ignored_decode_steps_target:
                    torch.cuda.synchronize()
                    ignored_decode_end = timed_decode_start = time.time()
                if collect_step_latency:
                    torch.cuda.synchronize()
                    step_start = time.time()
                logits = self._decode_step(output_ids, decode_step_idx)
                output_ids = self.sampling(logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
                if collect_step_latency:
                    torch.cuda.synchronize()
                    self.decode_step_latencies_ms.append((time.time() - step_start) * 1000)
                if not ignore_eos:
                    end_of_text |= torch.isin(output_ids, eos_tokens)
                    if end_of_text.all():
                        generated_ids[:, generated_count:generated_count + 1] = output_ids
                        generated_count += 1
                        print(colored("All sequences have reached EOS token, stop decoding.", 'yellow'))
                        break
                generated_ids[:, generated_count:generated_count + 1] = output_ids
                generated_count += 1

        if profile_timing:
            torch.cuda.synchronize()
            decode_end = time.time()
            decode_steps = max(generated_count - 1, 0)
            decode_duration = decode_end - decode_start
            ignored_decode_steps = min(max(int(ignore_first_steps_for_tpot), 0), decode_steps)
            timed_decode_steps = max(decode_steps - ignored_decode_steps, 0)
            if ignored_decode_steps > 0 and ignored_decode_start is not None:
                ignored_decode_stop = ignored_decode_end if ignored_decode_end is not None else decode_end
                ignored_decode_duration = ignored_decode_stop - ignored_decode_start
            else:
                ignored_decode_duration = 0.0
            if timed_decode_steps > 0:
                timed_start = timed_decode_start if "timed_decode_start" in locals() else decode_start
                timed_decode_duration = decode_end - timed_start
            else:
                timed_decode_duration = 0.0
            tpot_ms = decode_duration * 1000 / decode_steps if decode_steps else 0.0
            throughput = self.batch_size * decode_steps / decode_duration if decode_duration > 0 and decode_steps else 0.0
            end2end_duration = decode_end - end2end_start
            self.generate_latency_stats.update({
                "decode_s": decode_duration,
                "raw_decode_s": decode_duration,
                "ignored_decode_s": ignored_decode_duration,
                "timed_decode_s": timed_decode_duration,
                "ignored_decode_steps": ignored_decode_steps,
                "timed_decode_steps": timed_decode_steps,
                "tpot_ms": tpot_ms,
                "end2end_s": end2end_duration,
            })
            print(colored(
                f"Decoding latency: {round(decode_duration, 4)} s "
                f"({round(tpot_ms, 2)} ms/step), "
                f"Throughput: {round(throughput, 2)} tokens/s",
                'green'
            ))
            print(colored(f"TPOT latency: {round(tpot_ms, 2)} ms/token", 'green'))

            print(colored(f"End2End Latency: {round(end2end_duration, 4)} s\n", 'green'))
        
        outputs_tensor = generated_ids[:, :generated_count]
        if profile_timing:
            materialize_start = time.time()
        outputs_ids = outputs_tensor.tolist()
        if profile_timing:
            self.generate_latency_stats["materialize_s"] = time.time() - materialize_start
        
        return outputs_ids


    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config,
                 do_sample=False, temperature=0.6, top_p=0.95, top_k=20, ignore_eos=True, eos_token_ids=None,
                 prefill_bsz=1, prefill_method="full", collect_step_latency=False, profile_timing=False,
                 ignore_first_steps_for_tpot=0, use_cuda_graph=False):
        """ LLM Inference.
        Args:
            attention_type: str, Full_Flash_Attn or CometKV.
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
            attn_config (dict): The deoding attention configuration.
            do_sample, temperature, top_p, top_k, ignore_eos, eos_token_ids: The sampling parameters.
            prefill_bsz (int): The batch size for prefill.
            prefill_method (str): The method for prefill, support full and xattn.
            profile_timing (bool): Enable synchronized prefill, TPOT, and end-to-end latency logs.
        """
        self.attention_type = attention_type

        bs, input_length = inputs_ids.shape
        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        assert self.input_length + self.max_new_length <= self.max_length, \
            f"Error: input_length({self.input_length}) + max_new_length({self.max_new_length}) exceeds max_length({self.max_length})"

        # compute valid start position for each sequence
        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks

        self.prefill_bsz = min(prefill_bsz, self.batch_size)
        self.prefill_method = prefill_method
        if self.attention_type in ("CometKV", "CometKV_GPU") and not (valid_start == 0).all():
            raise ValueError("CometKV currently requires same-length inputs with valid_start == 0.")
        # set prefill batch size to 1 and prefill method to full attention if input sequences are not in the same length
        if not (valid_start == 0).all():
            self.prefill_bsz = 1
            self.prefill_method = "full"

        if profile_timing:
            print("Allocate GPU buffers and CPU pin memory ...")
        self.init_kv_cache(valid_start, attn_config)

        outputs = self.inference(
            inputs_ids, 
            do_sample=do_sample, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            ignore_eos=ignore_eos,
            eos_token_ids=eos_token_ids,
            collect_step_latency=collect_step_latency,
            profile_timing=profile_timing,
            ignore_first_steps_for_tpot=ignore_first_steps_for_tpot,
            use_cuda_graph=use_cuda_graph,
        )

        return outputs
