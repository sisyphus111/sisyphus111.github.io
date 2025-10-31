---
title: vllm是如何支持Eagle-3推测解码的？
---
### vLLM v1 中 EAGLE-3 推测解码全链路代码索引与说明

本文面向熟悉 Transformer 的读者，按“配置识别 → 执行接入 → 提案器（EAGLE-3 头）→ 模型实现 → KV 缓存配合 → 采样与结果整合”的顺序，梳理 v1 框架内 EAGLE-3 推测解码的全流程与关键代码位置，便于通读与二次开发。

---

## 一、配置识别与方法选择

- 入口：`SpeculativeConfig.__post_init__` 完成方法选择、自动识别与 EAGLE 配置替换。
  - 识别 `method in ('eagle','eagle3')`，或基于模型名自动检测：

`vllm/config.py`
```python
                # Automatically detect the method
                if self.method in ('eagle', 'eagle3'):
                    pass
                elif "eagle3-" in self.draft_model_config.model.lower():
                    # Auto-detect Eagle3 from model name
                    self.method = "eagle3"
                elif "eagle-" in self.draft_model_config.model.lower():
                    # Auto-detect Eagle (non-3) from model name
                    self.method = "eagle"
...
                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3"):
...
                    from vllm.transformers_utils.configs.eagle import (
                        EAGLEConfig)
...
                    self.draft_model_config.hf_config = eagle_config
```

- 说明：
  - 若未显式指定 `method`，会尝试从草稿模型名自动推断（包含 `eagle3-` → `eagle3`）。
  - 对 EAGLE/EAGLE-3，会将草稿模型的 HF 配置包装为内部的 `EAGLEConfig`，使后续模型加载与权重映射一致。

---

## 二、执行器接入与调度位置

- GPU 执行器在构造时接入 EAGLE 推测器（drafter），EAGLE-3 打开辅助隐状态输出：

`vllm/v1/worker/gpu_model_runner.py`
```python
        # Set up speculative decoding.
        if self.speculative_config and get_pp_group().is_last_rank:
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device,
                                             self)  # type: ignore
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = True
            elif self.speculative_config.method == "medusa":
                ...
            self.rejection_sampler = RejectionSampler()
```

- 推测阶段的调用链（在一次解码 step 内）：

`vllm/v1/worker/gpu_model_runner.py`
```python
            # 构造下一 token、截取目标 tokens/positions/hidden_states
            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
            )
            spec_token_ids = draft_token_ids.tolist()
```

- 说明：
  - EAGLE-3 时，主模型需输出多路辅助隐状态，以便 drafter 聚合（见下文“模型实现”与 `get_eagle3_aux_hidden_state_layers`）。

---

## 三、EAGLE 提案器（EagleProposer）

- 入口类：`vllm/v1/spec_decode/eagle.py::EagleProposer`
  - 接收主模型的 token/pos/hidden_states 与下一 token，构建 EAGLE 草稿输入，生成草稿序列（可多 token）。

`vllm/v1/spec_decode/eagle.py`
```python
class EagleProposer:
    def __init__(..., runner=None):
        self.runner = runner
        self.hidden_size = self.draft_model_config.get_hidden_size()
        ...
```

- EAGLE-3 聚合辅助隐状态、出草稿 token：

`vllm/v1/spec_decode/eagle.py`
```python
        if self.method == "eagle3":
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size
        # Shift ids / 替换最后 token 为 next_token_ids
```

- 关键注意点：
  - 调用 `runner.attn_metadata_builders[0].build(...)` 构建 per-layer 的注意力元数据，假设 EAGLE 头所有层位于同一 KVCacheGroup（当前位置共享）：

`vllm/v1/spec_decode/eagle.py`
```python
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        ...
        self.positions[:num_tokens] = target_positions
        self.hidden_states[:num_tokens] = target_hidden_states
```

- 多 token 草稿生成（仅 FlashAttention 后端支持此路径），包含越界位置信息处理与 slot mapping 更新：

`vllm/v1/spec_decode/eagle.py`
```python
        assert isinstance(attn_metadata, FlashAttentionMetadata)
        draft_token_ids_list = [draft_token_ids]
        ...
        exceeds_max_model_len = positions >= self.max_model_len
        clamped_positions = torch.where(exceeds_max_model_len, 0, positions)
        ...
        attn_metadata.slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)
        ...
        logits = self.model.compute_logits(last_hidden_states[:batch_size], None)
        draft_token_ids = logits.argmax(dim=-1)
        draft_token_ids_list.append(draft_token_ids)
        ...
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids
```

- 失败/回退窗口准备：当上一步草稿部分被拒绝，重新构造“应送入 drafter”的 token 窗口（扣除被拒绝个数）：

`vllm/v1/spec_decode/eagle.py`
```python
def prepare_inputs(...):
    # 根据 num_rejected_tokens 重算 query_start_loc 与 seq_lens
    # 返回新窗口 token_indices 与更新后的 CommonAttentionMetadata
    return spec_common_attn_metadata, token_indices
```

- EAGLE 头与主模型的“同组 KV Cache”校验：

`vllm/v1/spec_decode/eagle.py`
```python
def validate_same_kv_cache_group(...):
    assert len(set([... for layer_name in self.attn_layer_names])) == 1
```

---

## 四、EAGLE-3 头（Draft 模型）实现

- 文件：`vllm/model_executor/models/llama_eagle3.py`
  - 目标：在主模型层数之后，拼出 EAGLE-3 头（一个 LlamaDecoderLayer 变体 + 聚合 FC + 自身 LM Head）。
  - 关键类与接口：
    - `Eagle3LlamaForCausalLM`：对外的草稿模型入口；提供 `combine_hidden_states`、`compute_logits` 与词表映射。

`vllm/model_executor/models/llama_eagle3.py`
```python
class Eagle3LlamaForCausalLM(LlamaForCausalLM):
    def __init__(...):
        target_layer_num = vllm_config.model_config.get_num_layers(...)
        self.model = LlamaModel(..., start_layer_id=target_layer_num)
        self.lm_head = ParallelLMHead(self.config.draft_vocab_size, ...)
        self.draft_id_to_target_id = nn.Parameter(..., requires_grad=False)
...
    def compute_logits(...):
        # 将 draft 词表 logits 映射回 target 词表（若提供 d2t 映射）
...
    def combine_hidden_states(...):
        # 将多路辅助隐状态拼接后过 fc 做聚合
        return self.model.fc(hidden_states)
```
    
	- 重写的 `LlamaDecoderLayer`：将输入的“主模型 embed + hidden_states”拼接后做注意力与 MLP，适配 EAGLE-3 的跨层结构：

`vllm/model_executor/models/llama_eagle3.py`
```python
class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(...):
        # override qkv to accept concat(embeds, hidden)
    def forward(...):
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        hidden_states = self.self_attn(...)
        
        hidden_states, residual = self.post_attention_layernorm(...)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual
```

- 与主模型交互：主模型需输出辅助隐状态索引（EAGLE-3 需要多处层的 prenorm 隐状态）
  - Llama 主模型提供选择辅助隐状态层的接口：

`vllm/model_executor/models/llama.py`
```python
    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)
```
  - GPU 执行器在加载模型时，根据 `method=="eagle3"` 启用：

`vllm/v1/worker/gpu_model_runner.py`
```python
            if self.use_aux_hidden_state_outputs:
                self.model.set_aux_hidden_state_layers(
                    self.model.get_eagle3_aux_hidden_state_layers())
```

---

## 五、KV 缓存管理对 EAGLE 的配合（lookahead/重算隐藏态）

- 目的：为保证草稿模型能拿到“当前步最后一个 block 的隐藏态”，在命中 prefix cache 时主动丢弃最后一个匹配块，强制重算该块，从而得到需要的 hidden states。
- Full attention 情况：

`vllm/v1/core/single_type_kv_cache_manager.py`
```python
class FullAttentionManager(...):
    @classmethod
    def find_longest_cache_hit(..., use_eagle: bool):
        ...
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks
```
- Sliding window 情况：

`vllm/v1/core/single_type_kv_cache_manager.py`
```python
class SlidingWindowManager(...):
    @classmethod
    def find_longest_cache_hit(..., use_eagle: bool):
        ...
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks
```

---

## 六、拒绝采样与结果整合

- 组件：`RejectionSampler`（Triton 内核），将“草稿概率/ID + 目标 logits + bonus token”整合，输出最终 token。
  - 文件：`vllm/v1/sample/rejection_sampler.py`
  - 说明：EAGLE-3 当前实现按照议题说明默认使用 argmax 草稿（draft_probs 的持久化管理仍在迭代中）。

`vllm/v1/sample/rejection_sampler.py`
```python
class RejectionSampler(nn.Module):
    """
    按论文实现拒绝采样；支持贪心与随机两种路径
    """
```

---

## 七、注意事项与已知约束

- 草稿模型建议 `draft_tensor_parallel_size=1`；主模型可使用 TP。
- 多 token EAGLE-3 草稿生成路径目前绑定 FlashAttention 后端。
- KV 缓存需满足“EAGLE 层在同一 KVCacheGroup”假设（已在 `validate_same_kv_cache_group` 校验）。
- 与外部议题一致：EAGLE-3 在 v1 的支持路径已具备；“草稿概率缓存/传递”等优化仍按议题约定采用 argmax 暂行方案，可参考：
  - `[SpecDecode] Support EAGLE in V1 #15901`（`https://github.com/vllm-project/vllm/issues/15901`）