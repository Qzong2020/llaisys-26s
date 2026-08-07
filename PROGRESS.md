# Task 3 实施进度

## 最终验收目标

在 LLAISYS 的 C/C++ 后端实现 DeepSeek-R1-Distill-Qwen-1.5B（Qwen2 架构）的 CPU 文本生成：

- Python 层只负责读取配置、加载权重、管理模型生命周期和调用后端，不使用 Python 框架实现模型推理逻辑。
- 后端提供模型创建、销毁、权重加载/访问、推理和生成状态重置所需的 C API，并提供对应 ctypes 绑定。
- 实现带 KV Cache 的 prefill 和逐 token decode。
- 在 greedy/argmax 模式下，返回“输入 token + 新生成 token”的完整序列，并与 PyTorch 参考结果逐 token 完全一致。
- 不回归 Task 0～2 已有功能，最终通过：

  ```bash
  python test/test_infer.py --model /path/to/model --test
  ```

- 最终提交满足项目 CI、复现报告和平台状态说明要求。Task 3 的当前范围是 CPU；NVIDIA 推理属于 Task 4。

## 状态约定

- `未完成`：尚未开始实现，或尚未通过该子任务的独立验收。
- `已完成`：该子任务已经通过其独立验收。
- 每次只推进一个子任务；完成后先运行对应测试并检查 `git diff`，再更新本文件状态。

## 开发基线

- Task 2 官方 GitHub CI 已完整通过。
- Task 3 基于该官方 CI 已通过的版本继续开发，不重复执行完整 Task 2 基线测试。
- Task 3 修改到某个既有算子时，仍需运行该算子的官方测试和与本次修改直接相关的最小补充验证。

## 子任务

### T3-01 建立 Task 0～2 基线

- **目标**：确认当前 Tensor、Runtime 和 Task 2 算子在开始 Task 3 前的实际状态，为后续回归提供基线。当前直接采用已经完整通过的 Task 2 官方 GitHub CI 结果，不重复运行完整基线。
- **可能涉及的文件**：不修改源码；只运行 `test/test_runtime.py`、`test/test_tensor.py` 和 `test/ops/*.py`。
- **验收方法**：以 Task 2 官方 GitHub CI 完整通过的版本作为 Task 3 开发基线；后续只针对实际修改的既有模块运行对应官方测试，并在最终交付阶段执行完整回归。
- **当前状态**：已完成（Task 2 官方 GitHub CI 已完整通过）

### T3-02 补齐无 Bias Linear 调用能力

- **目标**：保证 Qwen2 中无 bias 的 O projection、MLP projection 和 LM head 可以安全调用 Linear；`None`/`nullptr` 不得被解引用。
- **可能涉及的文件**：
  - `src/llaisys/ops.cc`
  - `python/llaisys/ops.py`
  - 必要时检查 `src/ops/linear/op.cpp`
- **验收方法**：运行 `python test/ops/linear.py --device cpu`；另外独立验证 `bias=None` 的输出与 `torch.nn.functional.linear(..., bias=None)` 一致。
- **确认结果**：内部 `llaisys::ops::linear` 和 CPU kernel 已正确判断空 bias；问题位于外层边界。Python wrapper 原先会对 `None` 调用 `bias.lib_tensor()`，C API 适配层原先会对空句柄访问 `bias->tensor`。
- **实施结果**：Python wrapper 将 `None` 传为空 ctypes 指针；C API 适配层将空句柄转换为空 `tensor_t`，未修改内部 Linear 算法和官方测试。
- **验收结果**：
  - 重新构建及安装成功。
  - 原有 CPU Linear 官方测试完整通过，包括其全部 F32、F16、BF16 和大尺寸用例。
  - 额外的 F32 小尺寸 `bias=None` 验证通过，结果与 `torch.nn.functional.linear(..., bias=None)` 一致。
  - 当前环境运行测试时需通过 `LD_PRELOAD=/usr/lib/libstdc++.so.6` 使用与新构建共享库匹配的系统 `libstdc++`；这属于本地 Conda 运行库优先级问题，不是 Linear 数值或空指针问题。
- **当前状态**：已完成

### T3-03 建立 Qwen2 C++ 模型结构与构建目标

- **目标**：建立模型 Meta、权重所有权、临时张量、推理状态和惰性 KV Cache 的 C++ 数据结构，并将模型模块纳入 Xmake 构建。
- **可能涉及的文件**：
  - 新建 `src/models/qwen2/model.hpp`
  - 新建 `src/models/qwen2/model.cpp`
  - `xmake.lua`
- **验收方法**：`xmake` 和 `xmake install` 成功；使用小型测试 Meta 创建、销毁模型核心对象时不崩溃，资源所有权清晰且无明显泄漏。
- **当前状态**：未完成

### T3-04 实现 Qwen2 C API

- **目标**：实现模型创建、销毁、权重访问/加载和单步推理 API；增加显式 reset 能力，使同一模型对象可执行多次相互独立的生成。
- **可能涉及的文件**：
  - `include/llaisys/models/qwen2.h`
  - 新建 `src/llaisys/qwen2.cc`
  - `src/llaisys/llaisys_tensor.hpp`
- **验收方法**：共享库成功导出全部 Qwen2 符号；连续执行 create、weights、reset、destroy 不崩溃；非法参数能够被明确拒绝。
- **当前状态**：未完成

### T3-05 实现 Qwen2 ctypes 绑定

- **目标**：在 Python 底层绑定中准确镜像 C ABI，包括 Meta、Weights、模型句柄及各函数的 `argtypes`/`restype`。
- **可能涉及的文件**：
  - 新建 `python/llaisys/libllaisys/models/__init__.py`
  - 新建 `python/llaisys/libllaisys/models/qwen2.py`
  - `python/llaisys/libllaisys/__init__.py`
- **验收方法**：`import llaisys` 成功；Python 能创建模型句柄、取得权重结构、调用 reset 并安全销毁；ctypes 结构字段与 C 头文件逐项一致。
- **当前状态**：未完成

### T3-06 解析模型配置并创建后端模型

- **目标**：从模型目录读取配置，构造 `LlaisysQwen2Meta`，验证 Qwen2 必需参数及维度关系，并创建 CPU 模型实例。
- **可能涉及的文件**：
  - `python/llaisys/models/qwen2.py`
  - 必要时 `python/setup.cfg`
- **验收方法**：对目标模型正确解析 28 层、hidden size 1536、12 个 Q heads、2 个 KV heads、head dim 128、intermediate size 8960、词表 151936、BF16、RoPE theta、RMS epsilon、最大位置和 EOS；不支持或不一致的配置明确报错。
- **当前状态**：未完成

### T3-07 实现严格的 Safetensors 权重映射与加载

- **目标**：把 safetensors 中的全部权重复制到后端对应 Tensor，检查名称、层号、shape、dtype、重复项和遗漏项，不依赖名称顺序。
- **可能涉及的文件**：
  - `python/llaisys/models/qwen2.py`
  - `python/llaisys/libllaisys/models/qwen2.py`
  - 必要时 `python/setup.cfg`
- **验收方法**：目标模型的 339 个权重全部且仅加载一次：3 个全局权重，加上每层 12 个权重、共 28 层；任意未知、缺失、重复、shape 或 dtype 不匹配均明确失败；加载完成后源 safetensors 对象可以安全释放。
- **当前状态**：未完成

### T3-08 实现 Qwen2 Prefill 前向

- **目标**：在 C++ 后端完成输入 embedding、28 层 decoder、最终 RMSNorm、最后位置 LM head 和 argmax；每层包含 attention 与 MLP 两条残差路径。
- **可能涉及的文件**：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
  - 必要时相关算子实现，但不得进行与 Task 3 无关的重构
- **验收方法**：先执行 1 个新 token 的端到端检查；必要时使用 Tensor debug，逐层比较 embedding、norm、Q/K/V、RoPE、attention、MLP、最终 hidden state、logits 和 argmax，首个分叉点可被定位；最终第一个生成 token 与 PyTorch 一致。
- **当前状态**：未完成

### T3-09 实现惰性 KV Cache 与增量 Decode

- **目标**：prefill 将每层 K/V 写入缓存，后续 decode 每次只处理新 token；position id 和 causal attention 使用正确的历史长度；缓存按需增长且不超过模型最大位置。
- **可能涉及的文件**：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
  - 必要时 `src/llaisys/qwen2.cc`
- **验收方法**：prefill 后 cache length 等于 prompt length；后续每步输入 `ntoken == 1` 且 cache length 只增加 1；缓存推理和全序列重算得到相同下一 token；短序列不会分配最大 131072 长度的完整缓存。
- **当前状态**：未完成

### T3-10 实现 Python Greedy Generate 与生命周期管理

- **目标**：实现 `Qwen2.generate`，在每次新生成前 reset，返回输入和生成 token 的完整列表，遵守 `max_new_tokens` 和 EOS，并提供幂等资源释放。
- **可能涉及的文件**：
  - `python/llaisys/models/qwen2.py`
  - 必要时 `python/llaisys/models/__init__.py`
- **验收方法**：验证空生成、单 token prompt、普通 prompt、EOS 提前终止、同一模型连续两次生成和重复释放；Task 3 仅保证 top-k=1 的 greedy/argmax 行为，`top_p` 与 `temperature` 不参与 greedy 数值路径。
- **当前状态**：未完成

### T3-11 分阶段端到端一致性验证

- **目标**：从短生成逐步扩展到正式验收长度，保证完整 token 列表与 PyTorch 完全一致，并记录性能与内存表现。
- **可能涉及的文件**：原则上不修改测试；根据首个数值分叉点修正对应 Task 3 源码。
- **验收方法**：依次运行：

  ```bash
  python test/test_infer.py --model /path/to/model --test --max_steps 1
  python test/test_infer.py --model /path/to/model --test --max_steps 4
  python test/test_infer.py --model /path/to/model --test --max_steps 128
  ```

  三档完整 token 列表均与 PyTorch 完全一致；128 步运行使用增量 KV Cache。
- **当前状态**：未完成

### T3-12 完整回归与交付检查

- **目标**：确认 Task 3 没有破坏已有功能，没有修改测试来放宽要求，没有硬编码输出，也没有混入 Task 4 CUDA 工作。
- **可能涉及的文件**：不应新增功能性修改；检查全部 Task 3 改动、CI 配置和交付报告。
- **验收方法**：重新逐个运行 Task 0～2 测试及正式 Task 3 命令；检查 `git diff`；在 Linux/Windows CI 中构建通过；准备复现命令、输出 token、耗时、峰值内存和 CPU 平台状态说明。
- **当前状态**：未完成

## 重要风险

### 风险 1：BF16 Safetensors 无法直接通过 NumPy 加载

当前 Python 骨架使用：

```python
safetensors.safe_open(file, framework="numpy", device="cpu")
```

目标模型权重是 BF16，而当前环境通过 NumPy backend 读取 BF16 safetensors 会出现 `TypeError: data type 'bfloat16' not understood`。

处理方向：

- 优先评估使用 `framework="pt"` 仅取得 CPU BF16 张量的原始内存并立即复制到 LLAISYS 后端；不得使用 PyTorch 实现任何模型推理或数值逻辑。
- 如果项目规则禁止 Python 包装层出现任何 PyTorch 权重对象，则实现 safetensors header + mmap/raw bytes 加载路径。
- 无论采用哪种路径，都必须验证 dtype、shape、连续性和对象生命周期，避免指针失效或隐式 dtype 转换。

### 风险 2：KV Cache 不能按最大长度一次性分配

目标模型最大位置长度为 131072。若为 28 层、K/V 两份、2 个 KV heads、head dim 128 的 BF16 Cache 一次性按最大长度分配，缓存本身约需 3.5 GiB，叠加约 3.5 GB 模型权重和中间张量后，很可能超过本机或 CI 内存限制。

处理方向：

- KV Cache 惰性创建，按实际生成长度分配。
- 容量不足时采用受控扩容策略，并保留当前有效内容。
- 始终检查 `cache_length + ntoken <= max_position_embeddings`。
- 验收时记录短序列的实际 cache capacity 和峰值内存，确认没有隐式最大长度预分配。

## 当前已知测试覆盖缺口

以下仅记录问题，不修改现有测试：

1. `test/ops/linear.py` 当前官方用例只覆盖 `use_bias=True`，没有覆盖 Qwen2 大量使用的 `bias=None` 路径；T3-02 已通过一次独立的最小 PyTorch 对照验证，但官方测试覆盖缺口仍存在。
2. `test/ops/embedding.py` 调用了 `check_equal`，但没有对其返回值执行 `assert`，错误结果可能不导致测试失败。
3. `test/ops/argmax.py` 使用“最大值正确或最大索引正确”的 `or` 条件，没有要求二者同时正确。
4. 当前没有 Qwen2 模型创建/销毁、配置解析、权重完整性、reset、KV Cache 状态或多次 generate 的独立单元测试。
5. Task 3 的直接测试只有脚本式 `test/test_infer.py`；它最终检查完整 token 列表，但数值不一致时无法直接定位到具体层或算子。
6. `README_ZN.md` 在 Task 2 中提到的 `test/test_ops.py` 实际不存在；当前 CI 是逐个执行 `test/ops/*.py`。

后续可增加独立测试或开发期检查来覆盖上述路径，但不得修改现有断言以放宽验收标准。

## 当前总状态

- Task 3 实现：**未完成**
- Task 3 正式端到端验收：**未完成**
- 本文件创建时未实施 Task 3，也未修改任何源码或测试。
