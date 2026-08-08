import json
import math
from ctypes import byref, c_int, c_int64, c_size_t
from pathlib import Path
from typing import Sequence

from safetensors import safe_open

from ..libllaisys import DataType, DeviceType, LIB_LLAISYS, LlaisysQwen2Meta


_DTYPE_MAP = {
    "float16": DataType.F16,
    "float32": DataType.F32,
    "bfloat16": DataType.BF16,
}

_SAFETENSORS_DTYPE_MAP = {
    DataType.F16: "F16",
    DataType.F32: "F32",
    DataType.BF16: "BF16",
}

_TORCH_DTYPE_MAP = {
    DataType.F16: "torch.float16",
    DataType.F32: "torch.float32",
    DataType.BF16: "torch.bfloat16",
}

_GLOBAL_WEIGHT_FIELDS = {
    "model.embed_tokens.weight": "in_embed",
    "lm_head.weight": "out_embed",
    "model.norm.weight": "out_norm_w",
}

_LAYER_WEIGHT_FIELDS = {
    "input_layernorm.weight": "attn_norm_w",
    "self_attn.q_proj.weight": "attn_q_w",
    "self_attn.q_proj.bias": "attn_q_b",
    "self_attn.k_proj.weight": "attn_k_w",
    "self_attn.k_proj.bias": "attn_k_b",
    "self_attn.v_proj.weight": "attn_v_w",
    "self_attn.v_proj.bias": "attn_v_b",
    "self_attn.o_proj.weight": "attn_o_w",
    "post_attention_layernorm.weight": "mlp_norm_w",
    "mlp.gate_proj.weight": "mlp_gate_w",
    "mlp.up_proj.weight": "mlp_up_w",
    "mlp.down_proj.weight": "mlp_down_w",
}


def _positive_int(config, name):
    value = config.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"Qwen2 config field '{name}' must be a positive integer")
    return value


def _positive_float(config, name):
    value = config.get(name)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Qwen2 config field '{name}' must be a positive number")
    return float(value)


def _parse_meta(config):
    if config.get("model_type") != "qwen2":
        raise ValueError("Qwen2 config field 'model_type' must be 'qwen2'")

    architectures = config.get("architectures")
    if not isinstance(architectures, list) or "Qwen2ForCausalLM" not in architectures:
        raise ValueError("Qwen2 config must describe a Qwen2ForCausalLM architecture")

    if config.get("hidden_act") != "silu":
        raise ValueError("Qwen2 config field 'hidden_act' must be 'silu'")
    if config.get("use_sliding_window", False) is not False:
        raise ValueError("Qwen2 sliding-window attention is not supported")
    if config.get("use_mrope", False) is not False:
        raise ValueError("Qwen2 multimodal RoPE is not supported")
    if config.get("rope_scaling") is not None:
        raise ValueError("Qwen2 RoPE scaling is not supported")
    if config.get("tie_word_embeddings", False) is not False:
        raise ValueError("Qwen2 tied input/output embeddings are not supported")
    if config.get("attention_bias", True) is not True:
        raise ValueError("Qwen2 attention projections must include bias")
    if config.get("mlp_bias", False) is not False:
        raise ValueError("Qwen2 MLP bias is not supported")

    dtype_name = config.get("torch_dtype")
    if dtype_name not in _DTYPE_MAP:
        raise ValueError(f"Unsupported Qwen2 torch_dtype: {dtype_name!r}")

    nlayer = _positive_int(config, "num_hidden_layers")
    hs = _positive_int(config, "hidden_size")
    nh = _positive_int(config, "num_attention_heads")
    nkvh = _positive_int(config, "num_key_value_heads")
    di = _positive_int(config, "intermediate_size")
    maxseq = _positive_int(config, "max_position_embeddings")
    voc = _positive_int(config, "vocab_size")

    if hs % nh != 0:
        raise ValueError("Qwen2 hidden_size must be divisible by num_attention_heads")
    derived_dh = hs // nh
    dh = config.get("head_dim", derived_dh)
    if type(dh) is not int or dh <= 0:
        raise ValueError("Qwen2 config field 'head_dim' must be a positive integer")
    if hs != nh * dh:
        raise ValueError("Qwen2 hidden_size must equal num_attention_heads * head_dim")
    if nh % nkvh != 0:
        raise ValueError(
            "Qwen2 num_attention_heads must be divisible by num_key_value_heads"
        )

    end_token = config.get("eos_token_id")
    if type(end_token) is not int or end_token < 0 or end_token >= voc:
        raise ValueError(
            "Qwen2 config field 'eos_token_id' must be an integer within the vocabulary"
        )

    return LlaisysQwen2Meta(
        dtype=_DTYPE_MAP[dtype_name],
        nlayer=nlayer,
        hs=hs,
        nh=nh,
        nkvh=nkvh,
        dh=dh,
        di=di,
        maxseq=maxseq,
        voc=voc,
        epsilon=_positive_float(config, "rms_norm_eps"),
        theta=_positive_float(config, "rope_theta"),
        end_token=end_token,
    )


def _tensor_shape(tensor):
    ndim = LIB_LLAISYS.tensorGetNdim(tensor)
    shape = (c_size_t * ndim)()
    LIB_LLAISYS.tensorGetShape(tensor, shape)
    return tuple(shape)


def _weight_targets(weights, nlayer):
    targets = {
        name: getattr(weights, field) for name, field in _GLOBAL_WEIGHT_FIELDS.items()
    }
    for layer in range(nlayer):
        prefix = f"model.layers.{layer}."
        for suffix, field in _LAYER_WEIGHT_FIELDS.items():
            targets[prefix + suffix] = getattr(weights, field)[layer]

    expected_count = len(_GLOBAL_WEIGHT_FIELDS) + nlayer * len(_LAYER_WEIGHT_FIELDS)
    if len(targets) != expected_count or not all(targets.values()):
        raise RuntimeError("Qwen2 backend returned an incomplete weight structure")
    return targets


def _load_weights(model_path, model, meta):
    weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(model)
    if not weights_ptr:
        raise RuntimeError("Qwen2 backend returned a null weight structure")
    targets = _weight_targets(weights_ptr.contents, meta.nlayer)

    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights found in: {model_path}")

    dtype = DataType(meta.dtype)
    expected_safetensors_dtype = _SAFETENSORS_DTYPE_MAP[dtype]
    expected_torch_dtype = _TORCH_DTYPE_MAP[dtype]
    seen = {}
    load_plan = {file: [] for file in files}

    # Validate the complete set before writing any backend storage.
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as source:
            for name in source.keys():
                if name in seen:
                    raise ValueError(
                        f"Duplicate Qwen2 weight '{name}' in {seen[name]} and {file}"
                    )
                if name not in targets:
                    raise ValueError(f"Unknown Qwen2 weight '{name}' in {file}")

                view = source.get_slice(name)
                source_shape = tuple(view.get_shape())
                target = targets[name]
                target_shape = _tensor_shape(target)
                if source_shape != target_shape:
                    raise ValueError(
                        f"Qwen2 weight '{name}' shape mismatch: "
                        f"file has {source_shape}, backend expects {target_shape}"
                    )
                if view.get_dtype() != expected_safetensors_dtype:
                    raise ValueError(
                        f"Qwen2 weight '{name}' dtype mismatch: "
                        f"file has {view.get_dtype()}, backend expects "
                        f"{expected_safetensors_dtype}"
                    )
                if DataType(LIB_LLAISYS.tensorGetDataType(target)) != dtype:
                    raise RuntimeError(f"Qwen2 backend weight '{name}' has wrong dtype")
                if DeviceType(LIB_LLAISYS.tensorGetDeviceType(target)) != DeviceType.CPU:
                    raise RuntimeError(f"Qwen2 backend weight '{name}' is not on CPU")
                if not LIB_LLAISYS.tensorIsContiguous(target):
                    raise RuntimeError(f"Qwen2 backend weight '{name}' is not contiguous")

                seen[name] = file
                load_plan[file].append((name, target, source_shape))

    missing = set(targets) - set(seen)
    if missing:
        names = sorted(missing)
        preview = ", ".join(names[:5])
        if len(names) > 5:
            preview += f", ... ({len(names)} total)"
        raise ValueError(f"Missing Qwen2 weights: {preview}")

    # tensorLoad is synchronous, so each source tensor may be released immediately.
    for file, entries in load_plan.items():
        with safe_open(file, framework="pt", device="cpu") as source:
            for name, target, expected_shape in entries:
                tensor = source.get_tensor(name)
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(f"Qwen2 weight '{name}' changed shape while loading")
                if str(tensor.dtype) != expected_torch_dtype:
                    raise ValueError(f"Qwen2 weight '{name}' changed dtype while loading")
                if tensor.device.type != "cpu" or not tensor.is_contiguous():
                    raise ValueError(
                        f"Qwen2 weight '{name}' must be a contiguous CPU tensor"
                    )
                LIB_LLAISYS.tensorLoad(target, tensor.data_ptr())
                del tensor

    return len(seen)


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self._model = None
        self._model_path = Path(model_path)

        try:
            self._device = DeviceType(device)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported Qwen2 device: {device!r}") from exc
        if self._device != DeviceType.CPU:
            raise ValueError("Task 3 Qwen2 inference supports CPU only")

        if not self._model_path.is_dir():
            raise FileNotFoundError(
                f"Qwen2 model directory not found: {self._model_path}"
            )
        config_path = self._model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Qwen2 config not found: {config_path}")

        try:
            with config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Qwen2 config JSON: {config_path}") from exc
        if not isinstance(config, dict):
            raise ValueError("Qwen2 config root must be a JSON object")

        self._meta = _parse_meta(config)
        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(self._meta), self._device, device_ids, 1
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 backend model")
        self._loaded_weight_count = _load_weights(
            self._model_path, self._model, self._meta
        )

    def __del__(self):
        model = getattr(self, "_model", None)
        lib = globals().get("LIB_LLAISYS")
        if model is not None and lib is not None:
            lib.llaisysQwen2ModelDestroy(model)
            self._model = None

    def close(self):
        model = getattr(self, "_model", None)
        if model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if self._model is None:
            raise RuntimeError("Qwen2 model has been closed")
        if type(top_k) is not int or top_k != 1:
            raise ValueError("Task 3 Qwen2 generation supports top_k=1 only")
        if max_new_tokens is not None and (
            type(max_new_tokens) is not int or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer or None")

        try:
            prompt = list(inputs)
        except TypeError as exc:
            raise TypeError("Qwen2 inputs must be a sequence of token IDs") from exc
        for token in prompt:
            if type(token) is not int or token < 0 or token >= self._meta.voc:
                raise ValueError(
                    f"Qwen2 token IDs must be integers in [0, {self._meta.voc})"
                )
        if len(prompt) > self._meta.maxseq:
            raise ValueError(
                f"Qwen2 input length exceeds the maximum of {self._meta.maxseq}"
            )

        remaining = self._meta.maxseq - len(prompt)
        steps = remaining if max_new_tokens is None else min(max_new_tokens, remaining)
        if not prompt and steps > 0:
            raise ValueError("Qwen2 generation requires a non-empty prompt")

        # top_p and temperature intentionally do not affect the Task 3 greedy path.
        del top_p, temperature
        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)

        output = prompt.copy()
        for step in range(steps):
            infer_tokens = prompt if step == 0 else output[-1:]
            token_array = (c_int64 * len(infer_tokens))(*infer_tokens)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model, token_array, len(infer_tokens)
                )
            )
            output.append(next_token)
            if next_token == self._meta.end_token:
                break

        return output
