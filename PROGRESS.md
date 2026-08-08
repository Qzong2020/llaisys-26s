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
- **实施结果**：
  - 新增内部 `llaisys::models::qwen2::Model`，保存 Meta、设备、权重、临时 Buffer 和 KV Cache 状态；未新增或修改公共 C API。
  - 权重通过现有 `Tensor::create` 和 `shared_ptr` 管理，按 Qwen2 的 3 个全局权重及每层 12 个权重建立正确 shape；本阶段只分配权重存储，不实现权重文件映射或加载。
  - KV Cache 为每层保留空的 K/V tensor 槽位，初始 length/capacity 均为 0；未按 `maxseq` 分配缓存，也未实现后续扩容逻辑。
  - 新增 `llaisys-models` 静态构建目标，并作为共享库依赖；未实现 forward、Python binding、CUDA 或生成逻辑。
- **验收结果**：
  - `xmake` 成功，`llaisys-models` 编译、归档并链接进整体构建依赖图。
  - `xmake install` 成功。
  - 一次性 C++ smoke test 链接构建生成的 `libllaisys-models.a`，使用 2 层 F32 小型 Meta 成功创建并销毁模型；全部逐层权重容器数量以及全局权重、Q/K 和 MLP 关键 shape 断言通过；cache length/capacity 均为 0；进程退出码为 0。
  - `git diff --check` 通过。
- **当前状态**：已完成

### T3-04 实现 Qwen2 C API

- **目标**：实现模型创建、销毁、权重访问/加载和单步推理 API；增加显式 reset 能力，使同一模型对象可执行多次相互独立的生成。
- **可能涉及的文件**：
  - `include/llaisys/models/qwen2.h`
  - 新建 `src/llaisys/qwen2.cc`
  - `src/llaisys/llaisys_tensor.hpp`
- **验收方法**：共享库成功导出全部 Qwen2 符号；连续执行 create、weights、reset、destroy 不崩溃；非法参数能够被明确拒绝。
- **实施结果**：
  - 实现 create、destroy、weights、reset 和 infer 五个 C API 入口；Task 3 create 当前明确限定为单 CPU 设备，不静默接受尚未支持的多设备或 NVIDIA 请求。
  - C API 模型对象持有内部 C++ Model，并为内部 `tensor_t` 建立 C ABI 借用句柄；句柄随模型统一销毁，调用方可通过现有 `tensorLoad` 写入权重，但不能单独 `tensorDestroy`。
  - reset 转发给内部模型并清空推理 Buffer/Cache length 状态，可重复调用。
  - infer 入口和参数校验已存在，但完整 forward 属于 T3-08；本阶段调用 infer 会明确抛出 `Unimplemented function`，没有返回固定或伪造 token。
  - 未实现 ctypes、Python 模型包装、权重名称映射、forward、KV Cache 扩容或 CUDA。
- **验收结果**：
  - `xmake` 和 `xmake install` 成功。
  - 共享库导出 `llaisysQwen2ModelCreate`、`Destroy`、`Weights`、`Reset`、`Infer` 全部五个符号。
  - 一次性纯 C API C++ smoke test 成功完成 create、取得权重、检查全局/Q/MLP 关键 shape、连续两次 reset、destroy 和空指针 destroy；进程退出码为 0。
  - 空 Meta、多设备和非 CPU create 请求均按预期抛出 `std::invalid_argument`；infer 按预期明确报告未实现。
  - 借用的输入 embedding 权重句柄通过 `tensorLoad` 写入 F32 数据并由 `tensorGetData` 原样读回，随后随模型安全销毁。
  - 未修改任何官方测试；`git diff --check` 通过。
- **当前状态**：已完成

### T3-05 实现 Qwen2 ctypes 绑定

- **目标**：在 Python 底层绑定中准确镜像 C ABI，包括 Meta、Weights、模型句柄及各函数的 `argtypes`/`restype`。
- **可能涉及的文件**：
  - 新建 `python/llaisys/libllaisys/models/__init__.py`
  - 新建 `python/llaisys/libllaisys/models/qwen2.py`
  - `python/llaisys/libllaisys/__init__.py`
- **验收方法**：`import llaisys` 成功；Python 能创建模型句柄、取得权重结构、调用 reset 并安全销毁；ctypes 结构字段与 C 头文件逐项一致。
- **实施结果**：
  - 新增底层 `libllaisys.models.qwen2` 模块，逐项镜像 `LlaisysQwen2Meta` 和 `LlaisysQwen2Weights`，并将 opaque model 定义为 `c_void_p`。
  - 为 create、destroy、weights、reset、infer 五个现有 C API 设置完整 `argtypes`/`restype`，在加载共享库后统一注册。
  - 权重结构中的逐层字段使用 `POINTER(llaisysTensor_t)` 对应 C 端 `llaisysTensor_t *`；Python 端只读取这些借用句柄，不接管或单独销毁其所有权。
  - 未修改高层 `Qwen2` 类，未开始配置解析、权重文件映射、forward、KV Cache 或 generate。
- **验收结果**：
  - 官方 `python test/test_runtime.py --device cpu` 通过，`import llaisys` 和原有 runtime 路径未回归。
  - 一次性 Python ctypes smoke test 使用 2 层 F32 小型 Meta 成功完成 create、weights、关键 tensor shape 检查、连续两次 reset 和 destroy；进程退出码为 0。
  - 一次性 C++/Python ABI 探针逐字段比较结构大小与 offset：Meta 均为 88 字节，Weights 均为 120 字节，全部字段偏移一致。
  - 未修改任何官方测试；`git diff --check` 通过。
- **当前状态**：已完成

### T3-06 解析模型配置并创建后端模型

- **目标**：从模型目录读取配置，构造 `LlaisysQwen2Meta`，验证 Qwen2 必需参数及维度关系，并创建 CPU 模型实例。
- **可能涉及的文件**：
  - `python/llaisys/models/qwen2.py`
  - 必要时 `python/setup.cfg`
- **验收方法**：对目标模型正确解析 28 层、hidden size 1536、12 个 Q heads、2 个 KV heads、head dim 128、intermediate size 8960、词表 151936、BF16、RoPE theta、RMS epsilon、最大位置和 EOS；不支持或不一致的配置明确报错。
- **实施结果**：
  - 高层 `Qwen2` 从模型目录的 `config.json` 读取配置，映射 dtype、层数、hidden/Q/KV/head/intermediate 维度、最大位置、词表、RMS epsilon、RoPE theta 和 EOS 到 `LlaisysQwen2Meta`。
  - `head_dim` 缺省时由 `hidden_size / num_attention_heads` 推导；实现没有硬编码目标模型的 28 层或其他固定维度。
  - 在进入 C ABI 前检查 Qwen2 CausalLM 架构、SILU、dtype、正数类型、hidden/head 关系、GQA 整除关系和 EOS 范围；对当前后端不支持的 sliding window、M-RoPE、RoPE scaling、tied embeddings、无 attention bias 和 MLP bias 明确报错。
  - 构造函数当前只允许 Task 3 的单 CPU 路径，创建并持有后端模型；增加最小析构清理以避免本阶段创建的后端资源泄漏，完整公开生命周期与 generate 行为仍属于 T3-10。
  - 删除原构造函数中未实现的 safetensors 空循环；本阶段没有读取或加载任何权重数据，BF16 加载风险留待 T3-07 处理。
- **验收结果**：
  - 使用真实 `DeepSeek-R1-Distill-Qwen-1.5B/config.json` 成功解析出 28、1536、12、2、128、8960、131072、151936、BF16、1e-6、10000 和 EOS 151643，并成功创建、销毁 CPU 后端模型。
  - 通过后端借用权重句柄确认输入 embedding、最后一层 Q/K 和 MLP down 权重 shape 分别为 `(151936, 1536)`、`(1536, 1536)`、`(256, 1536)` 和 `(1536, 8960)`。
  - 9 类错误配置以及 NVIDIA 设备、缺失模型目录共 11 个反向场景均被明确拒绝。
  - 官方 `python test/test_runtime.py --device cpu` 通过；Python 语法编译检查和 `git diff --check` 通过。
  - 未修改任何官方测试；未开始权重映射、forward、KV Cache 或 generate。
- **当前状态**：已完成

### T3-07 实现严格的 Safetensors 权重映射与加载

- **目标**：把 safetensors 中的全部权重复制到后端对应 Tensor，检查名称、层号、shape、dtype、重复项和遗漏项，不依赖名称顺序。
- **可能涉及的文件**：
  - `python/llaisys/models/qwen2.py`
  - `python/llaisys/libllaisys/models/qwen2.py`
  - 必要时 `python/setup.cfg`
- **验收方法**：目标模型的 339 个权重全部且仅加载一次：3 个全局权重，加上每层 12 个权重、共 28 层；任意未知、缺失、重复、shape 或 dtype 不匹配均明确失败；加载完成后源 safetensors 对象可以安全释放。
- **实施结果**：
  - 建立标准 Qwen2 权重名到 C ABI 借用句柄的完整映射：3 个全局权重以及按 `nlayer` 生成的每层 12 个 attention/MLP 权重，不依赖 safetensors 中的键顺序或固定层数。
  - 使用两阶段加载：第一阶段通过 safetensors slice 元数据检查全部分片的未知、重复、缺失、shape、dtype，以及后端 dtype/device/contiguous 状态；只有完整集合全部通过后，第二阶段才写入后端，避免失败时产生半加载模型。
  - BF16 使用 `safe_open(..., framework="pt", device="cpu")` 取得连续 CPU Tensor 的原始数据指针，并立即调用同步 `tensorLoad` 按字节复制；没有使用 PyTorch 执行模型推理、数值计算或 dtype 转换，源 Tensor 可在每次复制后立即释放。
  - 构造完成后记录实际加载权重数；任意分片缺失、重复键、额外键、错误 shape 或错误 dtype 都会明确失败。
  - 未修改 C/C++ forward、KV Cache、generate 或 CUDA 实现。
- **验收结果**：
  - 一次性 1 层 F32 小模型跨两个乱序分片成功加载全部 15 个权重；逐权重比较后端与源 Tensor 的完整原始字节，全部一致。
  - 同一小模型分别验证重复、未知、缺失、shape 不匹配和 dtype 不匹配五类错误，全部按预期拒绝。
  - 真实 DeepSeek-R1-Distill-Qwen-1.5B 成功加载且仅加载 339 个 BF16 权重，源文件键集合与后端目标集合完全相同。
  - 对 embedding、最终 norm、不同层 Q/K/MLP 以及 LM head 的源/后端首部、中部和尾部原始 BF16 字节抽样比较全部一致；模型随后安全销毁。
  - 官方 `python test/test_runtime.py --device cpu` 通过；Python 语法编译检查和 `git diff --check` 通过。
  - 未修改任何官方测试；未调用尚未实现的 infer，也未开始 T3-08。
- **当前状态**：已完成

### T3-08 实现 Qwen2 Prefill 前向

- **目标**：在 C++ 后端完成输入 embedding、28 层 decoder、最终 RMSNorm、最后位置 LM head 和 argmax；每层包含 attention 与 MLP 两条残差路径。
- **可能涉及的文件**：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
  - 必要时相关算子实现，但不得进行与 Task 3 无关的重构
- **验收方法**：先执行 1 个新 token 的端到端检查；必要时使用 Tensor debug，逐层比较 embedding、norm、Q/K/V、RoPE、attention、MLP、最终 hidden state、logits 和 argmax，首个分叉点可被定位；最终第一个生成 token 与 PyTorch 一致。
- **实施结果**：
  - `Model::infer` 现在执行独立的全序列 prefill：输入 embedding、每层 attention RMSNorm、Q/K/V projection、RoPE、GQA causal attention、O projection 与第一条残差、MLP RMSNorm、gate/up projection、SwiGLU、down projection 与第二条残差，最后对末位置执行 final RMSNorm、LM head 和 argmax。
  - C API `llaisysQwen2ModelInfer` 已转发到内部模型前向，不再返回未实现；未新增公共 API。
  - 在 embedding 前检查非空输入、最大序列长度和每个 token 的词表范围，避免越界读取 embedding。
  - 临时 Buffer 按本次 `ntoken` 精确分配，并在下一次独立 prefill 开始时释放重建；最终 norm/logits 只计算最后一个位置。
  - 仅组合已有 CPU 算子，没有修改 Task 2 算法；本阶段不创建、不写入 KV Cache，cache length/capacity 仍为 0，也未实现 Python generate 或 CUDA。
- **验收结果**：
  - `xmake` 和 `xmake install` 成功，新共享库包含内部 `Model::infer`。
  - 一次性全零 C API smoke test 成功走完整前向并返回 argmax 0，确认 Python 实际加载的是新共享库。
  - 一次性 2 层 F32 小模型分别使用 1、3、4 token 做完整 prefill，三次 argmax 均与手写 PyTorch Qwen2 参考前向一致，覆盖多 token causal mask、GQA、RoPE、两条残差和末位置 LM head。
  - 真实 DeepSeek-R1-Distill-Qwen-1.5B 对输入 token `[100]` 做单 token prefill：Hugging Face 与 LLAISYS 的下一 token 均为 `608`；本机 LLAISYS 权重加载约 4.26 秒，未优化 CPU prefill 约 39.42 秒。
  - 官方 `python test/test_runtime.py --device cpu` 通过；`git diff --check` 通过。
  - 未修改任何官方测试和既有算子；完整 `test_infer.py` 仍需 T3-09 KV Cache、T3-10 generate 和 T3-11 分阶段验收，本阶段未提前执行这些里程碑。
- **当前状态**：已完成

### T3-09 实现惰性 KV Cache 与增量 Decode

- **目标**：prefill 将每层 K/V 写入缓存，后续 decode 每次只处理新 token；position id 和 causal attention 使用正确的历史长度；缓存按需增长且不超过模型最大位置。
- **可能涉及的文件**：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
  - 必要时 `src/llaisys/qwen2.cc`
- **验收方法**：prefill 后 cache length 等于 prompt length；后续每步输入 `ntoken == 1` 且 cache length 只增加 1；缓存推理和全序列重算得到相同下一 token；短序列不会分配最大 131072 长度的完整缓存。
- **实施结果**：
  - 首次 infer 作为 prefill 接收完整 prompt；cache 已有历史时严格要求 `ntoken == 1`，并检查 `cache_length + ntoken` 不超过模型最大位置。
  - position id 从当前 cache length 开始；每层把经过 RoPE 的新 K 和原始新 V 写入缓存，然后以新 Q 对 `[0, cache_length + ntoken)` 的完整 K/V 前缀执行 causal attention。
  - KV Cache 初始不分配存储；需要容量时从 1 开始按 2 倍受控扩展并封顶 `maxseq`，扩容只复制当前有效前缀，所有层新缓存准备完成后再统一替换旧缓存。
  - 只有整次 forward 和 argmax 成功后才更新 cache length；中途失败不会提交新的逻辑长度。
  - reset 清零 cache length 并保留已分配 capacity，允许后续生成复用内存；没有按 131072 最大长度预分配。
  - 未新增公共 API，未修改 attention 或其他 Task 2 算子，未实现 Python generate 或 CUDA。
- **验收结果**：
  - `xmake` 成功；构建产物、安装到 `lib/` 的共享库及最终 Python 包内共享库 SHA-256 一致。
  - 一次性内部 C++ 状态测试验证：初始 length/capacity 为 0；3-token prefill 后为 3/4；两次单 token decode 后依次为 4/4、5/8；多 token decode 被拒绝；reset 后为 0/8，随后可重新 prefill。
  - 一次性 2 层 F32 小模型连续生成 5 步，缓存路径得到 `[4, 16, 4, 16, 4]`，每一步均与 reset 后完整序列重算以及 PyTorch 参考一致；reset 后相同 prompt 恢复相同首 token。
  - 真实 DeepSeek-R1-Distill-Qwen-1.5B 对输入 `[100]` 验证 prefill 加一次 decode：Hugging Face 和 LLAISYS 均得到 `[608, 320]`；本机 prefill 约 41.88 秒，单 token decode 约 40.83 秒。
  - 官方 `python test/test_runtime.py --device cpu` 通过；`git diff --check` 通过。
  - 未修改任何官方测试；完整 Python generate 和正式 `test_infer.py` 留在 T3-10/T3-11，本阶段未提前实现。
- **当前状态**：已完成

### T3-10 实现 Python Greedy Generate 与生命周期管理

- **目标**：实现 `Qwen2.generate`，在每次新生成前 reset，返回输入和生成 token 的完整列表，遵守 `max_new_tokens` 和 EOS，并提供幂等资源释放。
- **可能涉及的文件**：
  - `python/llaisys/models/qwen2.py`
  - 必要时 `python/llaisys/models/__init__.py`
- **验收方法**：验证空生成、单 token prompt、普通 prompt、EOS 提前终止、同一模型连续两次生成和重复释放；Task 3 仅保证 top-k=1 的 greedy/argmax 行为，`top_p` 与 `temperature` 不参与 greedy 数值路径。
- **实施结果**：
  - `Qwen2.generate` 在每次有效生成前调用后端 reset；首步把完整 prompt 交给 prefill，后续每步只把上一个新 token 交给增量 decode，并返回“原输入 + 新 token”的新列表。
  - 生成步数受 `max_new_tokens` 和剩余 `max_position_embeddings` 共同约束；`max_new_tokens=None` 时使用剩余上下文容量，生成到 EOS 时立即停止。
  - Task 3 明确只接受 `top_k=1` 的 greedy 路径；`top_p` 和 `temperature` 不参与数值计算。非法 token、空 prompt 的非零生成、非法步数、超长 prompt 和关闭后的调用均在进入 C infer 前明确拒绝。
  - 新增幂等 `close()`；首次调用销毁后端模型并清空句柄，重复调用不再销毁，析构仍对未显式关闭的对象执行兜底释放。
  - 未修改 C/C++ forward、KV Cache、官方测试或 CUDA，也未提前执行 T3-11 的正式 `test_infer.py`。
- **验收结果**：
  - Python 语法编译检查通过。
  - 一次性 1 层 F32 小模型集成测试中，普通 3-token prompt 连续三次 generate 均与直接 C API greedy 循环的完整 token 列表一致，确认每次 reset 后结果可复现；单 token prompt 同样一致。
  - 同一测试验证了 `max_new_tokens=0`（含空输入）、EOS 首 token 提前终止、请求步数超过剩余上下文时封顶、`max_new_tokens=None` 使用剩余容量，以及不同 `top_p`/`temperature` 不改变 greedy 输出。
  - 空 prompt 的非零生成、越界/非整数 token 和 `top_k != 1` 均按预期拒绝；连续两次 `close()` 不崩溃，关闭后 generate 明确失败。
  - 官方 `python test/test_runtime.py --device cpu` 通过；`git diff --check` 通过。
  - 未修改任何官方测试；正式目标模型逐 token 对照属于 T3-11，本阶段未提前执行。
- **当前状态**：已完成

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
- **实施结果**：
  - 使用当前 Task 3 源码重新构建并安装 release 版本；构建目录、`lib/` 和 Python 包内的 `libllaisys.so` SHA-256 完全一致，确保三档测试使用同一最终二进制。
  - 按 1、4、128 的顺序运行未修改的官方 `test/test_infer.py`，全部使用目标 `DeepSeek-R1-Distill-Qwen-1.5B`、CPU 和 `--test` greedy 参数。
  - 三档均未出现 token 分叉，因此本阶段没有修改模型源码、算子或官方测试。
- **验收结果**：
  - `--max_steps 1` 通过：完整列表一致，首个新 token 为 `91786`；HF 3.34 秒，LLAISYS 59.11 秒，观测峰值 RSS 7491.32 MiB。
  - `--max_steps 4` 通过：四个新 token 为 `[91786, 0, 358, 2776]`，完整列表一致；HF 1.85 秒，LLAISYS 84.41 秒，观测峰值 RSS 7450.14 MiB。
  - `--max_steps 128` 通过：HF 与 LLAISYS 的完整 90-token 列表逐项一致；模型在生成 81 个新 token 后输出 EOS `151643` 并按预期提前结束。HF 33.30 秒，LLAISYS 712.34 秒，观测峰值 RSS 7587.20 MiB。
  - 长档位覆盖 9-token prompt prefill 和连续增量 decode；生成过程没有按 131072 最大位置预分配完整 KV Cache，也没有 OOM。
  - 官方脚本仅报告其既有的 attention mask/pad token 警告，不影响 greedy 列表一致性断言；三个进程退出码均为 0。
- **当前状态**：已完成

### T3-12 完整回归与交付检查

- **目标**：确认 Task 3 没有破坏已有功能，没有修改测试来放宽要求，没有硬编码输出，也没有混入 Task 4 CUDA 工作。
- **可能涉及的文件**：不应新增功能性修改；检查全部 Task 3 改动、CI 配置和交付报告。
- **验收方法**：重新逐个运行 Task 0～2 测试及正式 Task 3 命令；检查 `git diff`；在 Linux/Windows CI 中构建通过；准备复现命令、输出 token、耗时、峰值内存和 CPU 平台状态说明。
- **本轮检查结果**：
  - 使用 release 配置重新执行 `xmake` 和 `xmake install` 成功；构建目录、`lib/` 和 Python 包内的 `libllaisys.so` SHA-256 均为 `fcc41f5db62abf78f421bec740d7367feb9ab2ac2d3a5e38db889dd93470cefa`。
  - Task 0 的 `python test/test_runtime.py --device cpu`、Task 1 的 `python test/test_tensor.py` 以及 Task 2 的 8 个 `test/ops/*.py` 官方 CPU 脚本全部通过；Task 2 大尺寸用例及 F32、F16、BF16 覆盖均完成。
  - 正式运行 `python test/test_infer.py --model /public/swiftllm/summer/models/DeepSeek-R1-Distill-Qwen-1.5B --test` 通过；HF 与 LLAISYS 的完整 90-token 列表逐项一致，首四个新 token 为 `[91786, 0, 358, 2776]`，最后一个 token 为 EOS `151643`。本轮 HF 用时 34.44 秒，LLAISYS 用时 640.97 秒。
  - 峰值内存沿用 T3-11 对同一正式 128 步档位的观测结果 7587.20 MiB；本轮环境不存在 `/usr/bin/time`，没有伪造新的峰值数据。
  - `git diff --check` 通过；`git diff --name-only -- test` 为空。完整改动审计未发现固定测试输入/输出或固定生成 token，未修改、删除或放宽官方测试，未修改既有 Task 2 算子实现，也未加入 Task 4 CUDA 模型推理代码。
  - GitHub Actions 配置仍包含 `windows-latest` 和 `ubuntu-latest` release 构建矩阵。本机 Linux release 构建已通过；尝试以 `clang-cl` 配置 Windows 目标时，Xmake 因本机没有 MSVC/Windows SDK 而在源码编译前拒绝配置。当前 Task 3 改动尚未提交或推送，因此无法据实声明实际 Linux/Windows CI 已通过。
- **剩余验收项**：提交并推送当前 Task 3 改动后，确认 GitHub Actions 的 Ubuntu/Windows 两个 build job 均通过；在此之前不把本阶段标记为完成。
- **当前状态**：未完成（本地完整回归与 Linux 构建已通过，等待实际 Linux/Windows CI）

## 重要风险

### 风险 1：BF16 Safetensors 无法直接通过 NumPy 加载（T3-07 已处理）

当前 Python 骨架使用：

```python
safetensors.safe_open(file, framework="numpy", device="cpu")
```

目标模型权重是 BF16，而当前环境通过 NumPy backend 读取 BF16 safetensors 会出现 `TypeError: data type 'bfloat16' not understood`。

处理结果：

- T3-07 不再使用 NumPy backend，改用 `framework="pt"` 仅取得 CPU BF16 Tensor 的 dtype、shape、连续性和原始数据指针。
- 在同步 `tensorLoad` 返回前保持源 Tensor 有效，复制后立即释放；没有执行 Python/PyTorch 推理、数值计算或 dtype 转换。
- 真实模型 339 个权重已全部加载，并通过多位置原始 BF16 字节对照验证。

### 风险 2：KV Cache 不能按最大长度一次性分配（T3-09 已处理）

目标模型最大位置长度为 131072。若为 28 层、K/V 两份、2 个 KV heads、head dim 128 的 BF16 Cache 一次性按最大长度分配，缓存本身约需 3.5 GiB，叠加约 3.5 GB 模型权重和中间张量后，很可能超过本机或 CI 内存限制。

处理结果：

- KV Cache 已实现惰性创建和受最大位置约束的 2 倍扩容，并在扩容时保留当前有效 K/V 前缀。
- 每次 infer 均检查剩余位置容量；短序列状态测试中 3-token prefill 只分配 capacity 4，增长到 5 token 时 capacity 才扩为 8。
- reset 只清零逻辑长度并复用已有小容量，不会隐式分配 131072 长度缓存。

## 当前已知测试覆盖缺口

以下仅记录问题，不修改现有测试：

1. `test/ops/linear.py` 当前官方用例只覆盖 `use_bias=True`，没有覆盖 Qwen2 大量使用的 `bias=None` 路径；T3-02 已通过一次独立的最小 PyTorch 对照验证，但官方测试覆盖缺口仍存在。
2. `test/ops/embedding.py` 调用了 `check_equal`，但没有对其返回值执行 `assert`，错误结果可能不导致测试失败。
3. `test/ops/argmax.py` 使用“最大值正确或最大索引正确”的 `or` 条件，没有要求二者同时正确。
4. 当前没有持久化的 Qwen2 模型创建/销毁、配置解析、权重完整性、prefill、reset、KV Cache 状态或多次 generate 单元测试；T3-03～T3-10 使用了一次性 C++/Python smoke test 验证内部模型、公开 C API、ctypes ABI、真实配置、完整权重加载、prefill、缓存 decode、Python greedy generate 和幂等释放，但这些检查尚未进入官方测试集。
5. Task 3 的直接测试只有脚本式 `test/test_infer.py`；它最终检查完整 token 列表，但数值不一致时无法直接定位到具体层或算子。
6. `README_ZN.md` 在 Task 2 中提到的 `test/test_ops.py` 实际不存在；当前 CI 是逐个执行 `test/ops/*.py`。

后续可增加独立测试或开发期检查来覆盖上述路径，但不得修改现有断言以放宽验收标准。

## 当前总状态

- Task 3 实现：**已完成**
- Task 3 正式端到端验收：**已完成**
- Task 3 完整回归与交付检查：**未完成（本地回归已通过，等待实际 Linux/Windows CI）**
- 已完成阶段：T3-01、T3-02、T3-03、T3-04、T3-05、T3-06、T3-07、T3-08、T3-09、T3-10、T3-11。
- 下一个未完成阶段：T3-12 完整回归与交付检查；本地验证已完成，剩余实际 Linux/Windows CI 状态确认。
