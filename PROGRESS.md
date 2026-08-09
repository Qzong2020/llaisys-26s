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
  - GitHub Actions 配置仍包含 `windows-latest` 和 `ubuntu-latest` release 构建矩阵。本机 Linux release 构建已通过；尝试以 `clang-cl` 配置 Windows 目标时，Xmake 因本机没有 MSVC/Windows SDK 而在源码编译前拒绝配置，当时尚不能据此声明实际 Windows CI 已通过。
- **Windows 兼容修复**：
  - Windows CI 的 MSVC `/EHc` 构建在 `src/llaisys/qwen2.cc` 报 C4297：`extern "C"` API 被默认视为不抛异常，但 create/weights/reset/infer 中的参数检查会抛出 `std::invalid_argument`，该警告因 `set_warnings("all", "error")` 被升级为错误。
  - 在公共头中增加 C++ 下展开为 `noexcept(false)`、其他语言下为空的 `LLAISYS_MAY_THROW`，并只标注确实可能抛异常的四个 Qwen2 C API；定义处同步使用 `noexcept(false)`。没有关闭 C4297、放宽 warnings-as-errors 或删除参数检查。
  - C++17 头文件 `-Werror` 语法检查通过；Linux release 重新构建和安装成功，`qwen2.cc` 已实际重编译。
  - 一次性 C++ smoke test 使用 `static_assert` 确认四个 API 均为可抛异常声明，并确认空 Meta 仍抛出、可捕获 `std::invalid_argument`；官方 runtime CPU 测试通过。
  - 官方 `test/test_infer.py --test --max_steps 1` 通过：HF 与 LLAISYS 完整列表一致，首个新 token 仍为 `91786`；HF 0.98 秒，LLAISYS 57.67 秒。
- **实际 CI 确认**：
  - 当前 `main`、本地 `origin/main` 和 `git ls-remote origin refs/heads/main` 均指向提交 `9eccc185cb1ae824409eafc39ee5055c45258bd7`（`task3 solve windows`），确认修复已经推送。
  - GitHub Actions `Build and test` run `31247929952`（run #5）状态为 `completed/success`，其 head SHA 与上述提交一致。
  - `Build (ubuntu-latest, release)` job `93079476676` 和 `Build (windows-latest, release)` job `93079476732` 均为 `completed/success`；两边的构建安装以及 Assignment-0～3 步骤全部成功。
- **当前状态**：已完成

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
- Task 3 完整回归与交付检查：**已完成（本地完整回归以及实际 Ubuntu/Windows CI 均通过）**
- 已完成阶段：T3-01、T3-02、T3-03、T3-04、T3-05、T3-06、T3-07、T3-08、T3-09、T3-10、T3-11、T3-12。
- Task 3 全部阶段均已完成。

# Task 4 实施进度

## 最终验收目标

按照 `README_ZN.md` 的 Task 4 要求，在不回归现有 CPU 路径和 Task 3 Qwen2 推理的前提下：

- 从 Nvidia、天数、摩尔、沐曦中选择至少两款真实 CUDA/类 CUDA 平台，不能只声明源码“理论兼容”。
- 通过 `--nv-gpu=y` 条件化编译设备 Runtime、CUDA 算子和所需设备资源；关闭该选项时不得编译、链接或探测 CUDA SDK。
- 完整实现 `LlaisysRuntimeAPI` 的设备、Stream、内存分配/释放、同步/异步拷贝接口。
- 为 Task 2 的 7 个正式算子及 Add 实现 F32、F16、BF16 CUDA 路径，并保持模型依赖的原地写和无 bias 行为正确。
- 复用 Task 3 已完成的 Qwen2 拓扑、权重映射、greedy generate 和惰性 KV Cache，仅把模型存储、前向和状态管理改造成设备安全的单设备 CUDA 路径；Task 4 不扩展多卡模型切分。
- 在两款目标平台上分别通过 Runtime、全部 CUDA 算子和以下正式推理命令，完整 token 列表与 PyTorch/reference 一致：

  ```bash
  python test/test_infer.py --model /path/to/model --test --device nvidia
  ```

- 最终完成 CPU/CUDA 回归、CI 与双平台实机复现报告，逐平台记录硬件、驱动、SDK/编译器、结果、耗时和峰值显存。

## 状态约定与执行顺序

- `未完成`：尚未开始实现，或尚未通过该子任务的全部独立验收。
- `已完成`：对应实现、官方测试、必要的严格补充验证及直接相关 CPU 回归均已通过。
- Task 4 分析和拆分已完成；T4-01 已冻结双平台（NVIDIA A800 + 沐曦 MetaX C500）并闭合 `rearrange` 范围（用户决定保持未实现、不计入 Task 4 范围）。
- 后续仍遵守一次只推进一个独立子任务；依赖顺序为 T4-01 → T4-02 → T4-03 → T4-04 → T4-05 → T4-06～T4-13 → T4-14 → T4-15 → T4-16 → T4-17 → T4-18 → T4-19。
- 每个硬件相关子任务必须保留两款平台各自的真实运行记录；第一款平台的结果不能替代第二款平台验收。

## 当前仓库基线与范围判断

- `xmake.lua` 已有 `--nv-gpu`、`ENABLE_NVIDIA_API` 和 `includes("xmake/nvidia.lua")`，但 `xmake/nvidia.lua` 尚不存在；当前 CUDA 开关无法形成完整构建。
- `src/device/nvidia/` 已有 Runtime/Resource 骨架，不应重复设计；Runtime 全部接口仍为占位实现，且 `memcpyAsync` 当前缺少公共 ABI 要求的 Stream 参数。
- Add、Argmax、Embedding、Linear、RMSNorm、RoPE、Self-Attention、SwiGLU 的派发层均已有 NVIDIA 分支，但仍是 `TO_BE_IMPLEMENTED()`，也没有 `src/ops/*/nvidia/` 实现。
- 当前公共设备枚举和 Python/官方测试只暴露 `LLAISYS_DEVICE_NVIDIA` / `--device nvidia`。T4-01 已实机证实平台 B（沐曦 MetaX C500，MACA cu-bridge）可复用同一逻辑 CUDA backend，仅需构建层适配（T4-02 将 CUDA 编译器从 nvcc 切换为 mxcc/cucc），不新增公共设备类型或厂商分支。
- 平台 A 固定为 Nvidia（node4 双 A800 80GB）；平台 B 已冻结为当前环境的沐曦 MetaX C500（MACA 3.5.3.20 + cu-bridge），有真实硬件、SDK、编译器、BLAS 与 PyTorch reference 实测证据。
- `Context` 当前存在 GPU 启用后必须验证的生命周期风险：`_current_runtime` 未显式初始化，`setDevice` 修改 Runtime vector 副本，且 Runtime 在激活目标设备前创建 Stream。
- Qwen2 C API、Python 构造和权重检查目前明确限制 CPU；KV Cache 扩容/写入使用 `std::memcpy`，最终 argmax 直接解引用 Tensor 地址，均不能用于设备指针。
- 当前 CUDA 官方测试存在覆盖缺口，不能利用这些缺口获得 PASS：Runtime 会在零设备时静默跳过；Embedding 没有断言比较结果；Argmax 只要求值或索引之一正确；Linear 不测无 bias；Add、RoPE、SwiGLU 不测模型使用的原地路径；Self-Attention 的 GPU reference mask 还需核对设备放置。
- `test/test_ops.py` 实际不存在，算子回归必须逐个运行现有 8 个 `test/ops/*.py` 脚本。
- `rearrange` 虽有公共入口和源码目录，但 README 没有定义其语义，CPU 仍未实现、没有官方测试且 Qwen2 不使用。第二轮调查经 GitHub API 全树搜索确认 InfiniTensor/InfiniLM 上游也不存在该算子；用户已于 2026-08-09 决定**保持未实现**，不作为 Task 4 范围，不猜测语义。Task 4 可审计算子范围确定为 Add + 7 个正式算子共 8 个。
- 当前开发节点 `node4` 的默认沙箱不暴露 `/dev/nvidia*`，但经获批的只读设备探针已确认宿主 NVIDIA Runtime 和两张 A800 均可实际运行；后续 GPU 验证必须使用同等级设备权限，不能把沙箱内零设备误报为平台状态。
- 2026-08-08 第二轮调查发现当前工作环境（容器 `cf50adb0c87a`）是可直接运行的 **沐曦 MetaX 平台 B 环境**：`/dev/mxcd` 设备节点存在，`mx-smi` 报告 1 张 MetaX C500，PyTorch `2.8.0+metax3.5.3.9` 的 `torch.cuda` 实测可用；MACA 3.5.3.20（`/opt/maca`）提供 mxcc 编译器与 cu-bridge CUDA 兼容层，`mxcc -x maca` 可编译并运行标准 CUDA kernel。该环境没有 NVIDIA 设备、没有 xmake、没有 github 外网，目标 `DeepSeek-R1-Distill-Qwen-1.5B` 模型也尚未可见，属于 T4-02/T4-15 前的平台 B 工程前置。
- T3-12 的实际 Ubuntu/Windows GitHub Actions 已确认通过，可作为 Task 4 开发基线。

## 子任务

### T4-01 冻结双平台、工具链与验收矩阵

- **目标**：确定 Nvidia 平台 A 和一款天数/摩尔/沐曦平台 B 的具体设备、驱动、SDK、编译器、BLAS 能力和 PyTorch/reference 运行方式；确认两平台是否共用现有 `LLAISYS_DEVICE_NVIDIA` ABI，并闭合 `rearrange` 是否属于官方 Task 4 范围。
- **依赖**：Task 4 规划完成；开始实现前确认 T3-12 的实际 CI 状态，或明确记录采用当前本地完整回归作为开发基线。
- **可能涉及的文件**：原则上不修改源码；更新 `PROGRESS.md`，交付阶段可同步平台报告。
- **验收方法**：形成包含两款真实平台的矩阵，逐项记录设备型号、可见设备数、驱动、Runtime/SDK、编译器、BLAS、PyTorch 设备映射、构建命令和算力有效期；两边均确认有至少一张可运行设备；记录逻辑 backend 复用或最小厂商适配结论；取得 `rearrange` 范围依据。平台 B 未确定或只有理论兼容说明时不得完成本任务。
- **本轮调查结果**：
  - Task 4 前置基线已闭合：当前远端 `main` 的 Task 3 提交 `9eccc185cb1ae824409eafc39ee5055c45258bd7` 对应 GitHub Actions run #5；Ubuntu/Windows release 两个 job 及 Assignment-0～3 全部通过，T3-12 已据实标记完成。
  - 平台 A 已冻结为当前 `node4` 的 NVIDIA A800 80GB PCIe：宿主共 2 张，compute capability 8.0，驱动 `550.144.03`，`nvidia-smi` 报告 driver-supported CUDA 12.4；CUDA Toolkit 为 12.8（`nvcc` 12.8.61，Runtime 12.8.57），cuBLAS 为 12.8.3.14，主机编译器为 GCC 14.2.1 / Clang 19.1.7，Xmake 为 3.0.9-dev。
  - 平台 A 的 reference 路径已实机验证：PyTorch `2.10.0+cu128` 通过标准 `torch.cuda` 看到 2 张 A800，CUDA 单元素加法成功；两张设备上的 F32、F16、BF16 `torch.mm` 均得到正确结果，BF16 capability 为 true；系统 CUDA Runtime 报 2 个设备，cuBLAS handle 可成功创建、查询和销毁。
  - 平台 A 后续构建命令沿用官方 `xmake f --nv-gpu=y -cv && xmake && xmake install`，逻辑设备映射为现有 `LLAISYS_DEVICE_NVIDIA` / `--device nvidia`。该命令属于 T4-02，T4-01 未提前修改构建或运行项目 CUDA 测试。
  - 本轮实机访问验证时间为 2026-08-08；当前 shell 不在 Slurm allocation 中，Slurm 也未给出账户算力到期时间，因此不能把本次可访问状态解释为后续阶段的有效期保证。
  - 平台 B 尚未冻结：本机没有发现天数、摩尔或沐曦设备节点、SDK、编译器、管理工具、BLAS 或 PyTorch 扩展；Slurm 的 6 个节点仅登记无厂商类型的 `gpu:2`，没有平台/SDK/有效期证据。一次 node6 的只读作业探针因 Slurm `Error generating job credential` 未能启动，且没有遗留作业；不能据此臆选第二平台。
  - 当前仓库的公共设备枚举、Runtime 派发、Python 和测试只定义 CPU/NVIDIA。平台 A 明确复用 `LLAISYS_DEVICE_NVIDIA`；平台 B 是否可复用该逻辑 ABI，必须在取得具体平台及兼容 SDK 后判断，目前不新增公共设备类型或厂商分支。
  - `README_ZN.md` 的 Task 2 明确列出 7 个正式算子，Task 4 CI 再覆盖 Add，共 8 个；全文没有定义 `rearrange`，仓库也没有其测试，Qwen2 不调用它。当前可审计范围因此是 Add 加 7 个正式算子，但 Task 4 的“每个算子目录”措辞仍有歧义；在取得官方/导师书面确认前，不猜测 `rearrange` 语义或实现它。
- **第二轮调查结果（2026-08-08，平台 B 证据冻结）**：
  - 平台 B 已冻结为当前工作环境中的 **沐曦 MetaX C500**：`/dev/mxcd` 设备节点存在，`mx-smi` 2.2.12 实测报告 1 张可见、GPU State Available、Kernel Mode Driver 3.8.30、BIOS 1.33.5.0；PyTorch 报告 cc (8,0)、SM 104、可见显存约 16.3 GB（板载 64 GB，切片配额）。
  - SDK 为 MACA 3.5.3.20（`/opt/maca` → `/opt/maca-3.5.3`），含 mxgpu_llvm/mccompiler/mcruntime/mcblas/mcblasLt/mcdnn/mccl/mcfft/mcsparse 等组件；编译器为 mxcc 1.0.0（clang-based，metaxgpu/xcore1000 后端）。MACA 自带 **cu-bridge** CUDA 兼容层（`/opt/maca-3.5.3/tools/cu-bridge/`）：提供 cuda_runtime.h、cublas_v2.h、cublasLt.h、cudnn.h 等完整 CUDA 头文件，`libcuda.so`→`libsymbol_cu.so` 导出 cudaMalloc/cudaMemcpy/cudaMemcpyAsync/cudaStreamCreate/cudaDeviceSynchronize 等 553 个 CUDA 符号，另有 `cucc` 编译包装脚本和 CMake 模块（FindCUDA/FindMACA）。
  - reference 路径实机验证：PyTorch `2.8.0+metax3.5.3.9` 的 `torch.cuda` 看到 1 张 MetaX C500；F32、F16、BF16 矩阵乘全部正确，`torch.cuda.is_bf16_supported()` 为 True。
  - ABI 复用实机验证：用 `mxcc -x maca -offload-arch native -I<cu-bridge/include> <src>.cu -o <out> --maca-path=/opt/maca -L/opt/maca/lib -lsymbol_cu -lruntime_cu` 编译标准 CUDA vector-add kernel，链接并实际运行成功（`1+2=3` 正确，进程退出码 0）。结论：**平台 B 可复用现有 `LLAISYS_DEVICE_NVIDIA` 逻辑 backend 与 CUDA 源码，只需最小构建层适配**（T4-02 将 CUDA 编译器从 nvcc 切换为 mxcc/cucc，使用 cu-bridge include 并链接 `libsymbol_cu`/`libruntime_cu`），无需新增公共设备类型或厂商分支。
  - 平台 B 工程约束：该环境没有 NVIDIA 设备（无 `/dev/nvidia*`、无 `nvidia-smi`、无 `/usr/local/cuda`）；没有 xmake 且无 github 外网（pip 走内网镜像 `mirrors.aliyun.com`）；目标模型 `DeepSeek-R1-Distill-Qwen-1.5B` 尚未可见（`/public/swiftllm` 不存在）。这三项需要在 T4-02/T4-15 前完成平台 B 的构建工具与模型部署，不阻塞 T4-01 冻结矩阵。
  - 平台 A 本轮状态：当前沙箱无法重新实机访问 NVIDIA（无 `/dev/nvidia*`）；沿用上一轮 node4 宿主探针证据（2×A800、`torch.cuda`/cuBLAS 三 dtype 实测通过）作为平台 A 冻结依据，T4-03 起的平台 A 实机验收仍需同等级设备权限。
  - `rearrange` 范围依据：官方 `README_ZN.md` 的 Task 4 全文未定义 `rearrange`；官方 CI（`.github/workflows/build.yaml`）的 Assignment-2 步骤只运行 add/argmax/embedding/linear/rms_norm/rope/self_attention/swiglu 共 8 个算子脚本，不含 `rearrange`；`rearrange` 无 CPU 实现、无官方测试、Qwen2 不调用，且经 GitHub API 全树搜索确认 InfiniTensor/InfiniLM 上游也不存在该算子。用户 2026-08-09 决定**保持未实现**、不计入 Task 4 范围；可审计范围为 Add 加 7 个正式算子共 8 个。
- **当前验收矩阵**：

  | 项目 | 平台 A | 平台 B |
  | --- | --- | --- |
  | 厂商/设备/数量 | NVIDIA A800 80GB PCIe × 2（node4） | 沐曦 MetaX C500 × 1（当前环境；板载 64 GB，切片约 16.3 GB） |
  | 驱动 | 550.144.03 | Kernel Mode Driver 3.8.30；`/opt/mxdriver` 3.5.3.11 |
  | Runtime / SDK | CUDA driver API 12.4；Toolkit / Runtime 12.8 | MACA 3.5.3.20（`/opt/maca`）+ cu-bridge CUDA 兼容层（`libsymbol_cu.so` 导出 553 个 CUDA 符号） |
  | 编译器 | nvcc 12.8.61；GCC 14.2.1；Clang 19.1.7；Xmake 3.0.9-dev | mxcc 1.0.0（`-x maca -offload-arch native`）；Xmake 尚未部署 |
  | BLAS | cuBLAS 12.8.3.14，已实际创建 handle | mcblas / mcblasLt 3.5.3.20（cu-bridge 提供 cublas_v2.h 兼容头） |
  | reference | PyTorch 2.10.0+cu128 / `torch.cuda`，2 卡三 dtype 实测通过 | PyTorch 2.8.0+metax3.5.3.9 / `torch.cuda`，1 卡三 dtype 实测通过 |
  | LLAISYS 逻辑 ABI | `LLAISYS_DEVICE_NVIDIA` / `--device nvidia` | 复用 `LLAISYS_DEVICE_NVIDIA`；已实机编译并运行 CUDA kernel 验证 |
  | 计划构建命令 | `xmake f --nv-gpu=y -cv && xmake && xmake install` | 同一命令 + mxcc/cu-bridge 构建适配（T4-02）；xmake 需先在平台 B 部署 |
  | 算力有效期 | 2026-08-08 当日可运行；未提供到期时间 | 2026-08-08 当日实测可用；未提供到期时间 |

- **阻塞项**：已全部闭合。
  - 平台 B 已由当前工作环境实机证据闭合（沐曦 MetaX C500，`torch.cuda` 与 CUDA kernel 均实测运行）。
  - `rearrange` 范围已由用户于 2026-08-09 决定：**保持未实现**、不计入 Task 4 范围（该算子无任何语义定义，见上）；不猜测实现。
  - 平台 B 的构建工具链（xmake）和目标模型部署属于 T4-02/T4-15 的工程前置，不属于 T4-01 验收内容。
- **当前状态**：已完成（双平台矩阵冻结、逻辑 backend 复用结论实机验证、`rearrange` 范围闭合）

### T4-02 接通可开关的双平台 CUDA 构建

- **目标**：建立 CUDA Runtime/算子静态目标及最终共享库的条件依赖和链接流程；复用现有 `--nv-gpu` 开关，使 CPU-only 构建完全不依赖 CUDA。
- **依赖**：T4-01。
- **可能涉及的文件**：
  - `xmake.lua`
  - 新建 `xmake/nvidia.lua`
  - 仅当 T4-01 证明必要时新增最小厂商构建适配文件
- **验收方法**：在两平台分别从干净 Xmake 配置执行 `xmake f --nv-gpu=y -cv && xmake && xmake install`，最终 Python 包能加载对应共享库；使用 `--nv-gpu=n` 从干净配置构建安装成功、无需 CUDA SDK，且 `python test/test_runtime.py --device cpu` 通过；检查不同平台构建缓存没有混用。此阶段只打通构建，不实现 Runtime 或算子算法。
- **本轮实现（2026-08-09，平台 B）**：
  - 平台 B 构建工具前置已闭合：通过 `mamba install -y -c conda-forge xmake`（tuna 清华 conda-forge 内网镜像）部署 xmake 3.0.9，位于 `/opt/conda/bin/xmake`；因沙箱以 root 运行，需 `export XMAKE_ROOT=y`。
  - 新建 `xmake/nvidia.lua`：核心是 `option("maca-cuda")` 的 `on_check`——当 `--nv-gpu=y` 且未显式 `--cuda=...` 时，检测 MACA cu-bridge（`mxcc` + `cu-bridge/include` + `cucc`），若存在则在 `$HOME/.cache/llaisys/maca-cuda/` 幂等地合成最小 "CUDA SDK" shim：`bin/nvcc` 为调用 `cucc` 的 bash 包装（cucc 自动加 `-x maca` 把 nvcc 风格参数转给 mxcc），`include/` 软链到 cu-bridge/include（满足 cucc 强制 `__macro_mxcc.h` 的包含路径），`lib64/libcudart_static.a`/`libcudadevrt.a` 为空归档（满足 xmake CUDA 规则追加的 `-lcudart_static -lcudadevrt` 链接行）。随后 `config.set("cuda", shim)` 让 xmake 内置 CUDA 机制用同一套机器驱动 mxcc/cucc。
  - `llaisys-device-nvidia` / `llaisys-ops-nvidia` 静态目标：编译 `src/device/nvidia/*.cu` 与 `src/ops/*/nvidia/*.cu`，`add_cuflags("-Xcompiler -fPIC", {force = true})` 保证 PIC（mxcc 自动旗标检查会误丢该参数）；`on_config` 仅当 `maca_cu_bridge` 时追加 `-L/opt/maca/lib -lsymbol_cu -lruntime_cu` 与 rpath，NVIDIA 平台自然不链接 MACA 库。
  - `xmake.lua`：`llaisys-device` 与 `llaisys-ops` 在 `has_config("nv-gpu")` 时条件依赖上述两个 NVIDIA 目标；`--nv-gpu=n` 完全不触碰 CUDA。
  - `src/device/nvidia/nvidia_runtime_api.cu`：修正 `memcpyAsync` ABI 签名（补 `llaisysStream_t stream` 第 5 参，与公共 `memcpy_async_api` 对齐），是 T4-02 的构建阻塞项；函数体仍为 `TO_BE_IMPLEMENTED()`，完整实现属于 T4-03。
- **平台 B 验证记录（2026-08-09）**：
  - `--nv-gpu=y` 干净构建：`xmake f --nv-gpu=y -cv && xmake && xmake install` 成功，`python/llaisys/libllaisys/libllaisys.so` 生成并被 Python 加载，`ldd` 解析到 `/opt/maca/lib/libruntime_cu.so`、`libmcruntime.so` 等 MACA 依赖；`python test/test_runtime.py --device cpu` 在 CUDA 构建下通过。
  - `--nv-gpu=n` 干净构建：重新 `xmake f` 后安装成功，**无需 CUDA SDK/MACA**；`test/test_runtime.py --device cpu` 通过；CPU 回归全部通过：`add`、`argmax`、`embedding`、`linear`、`rms_norm`、`rope`、`self_attention`、`swiglu`（8/8 算子）与 `test_tensor.py` 均 `Test passed!`。`ldd` 确认 CPU 构建无任何 MACA 依赖。
  - 缓存隔离：`build/` 与 `.xmake/` 均为各 checkout 本地、gitignore；两平台各自执行自己的 `xmake f`，未观察到配置混用。
- **平台 A 状态**：nvidia.lua 的 NVIDIA 路径走 xmake 内置 CUDA 机制（自动检测 `/usr/local/cuda*` 或 `--cuda=...` 驱动 nvcc），无需 shim；当前沙箱无 `/dev/nvidia*` 无法实机执行，平台 A 的 `--nv-gpu=y` 实机验收需 T4-03 起的同等级设备权限。
- **已知边界**：`--device nvidia` 运行 `test_runtime.py` 抛 `TO_BE_IMPLEMENTED`（预期，T4-03 前不实现 Runtime 算法）；CUDA 构建下 CPU 算子回归会因 `src/core/context/context.cpp:9-21` 的设备枚举调用 NVIDIA `getDeviceCount()` 而崩溃，已确认根因并归入 T4-05（Context/Runtime/Resource/Tensor 设备生命周期），T4-02 不越界修复。
- **当前状态**：已完成（平台 B 双开关均验证；平台 A 构建路径已实现、实机验收待设备权限）

### T4-03 实现 Nvidia CUDA Runtime API

- **目标**：在平台 A 完整实现设备查询/切换/同步、Stream 创建/销毁/同步、Device/Host 内存分配释放、四种方向的同步与异步拷贝，并统一检查 CUDA 调用错误；修正 `memcpyAsync` ABI 签名。
- **依赖**：T4-02。
- **可能涉及的文件**：
  - `src/device/nvidia/nvidia_runtime_api.cu`
  - `src/device/runtime_api.hpp`
  - `src/device/runtime_api.cpp`
  - `src/device/nvidia/nvidia_resource.cu`
  - `src/device/nvidia/nvidia_resource.cuh`
  - `xmake/nvidia.lua`
- **验收方法**：先运行官方 `python test/test_runtime.py --device nvidia`，并额外断言设备数大于 0；再用独立严格测试覆盖 `malloc/free device`、`malloc/free host`、H2H/H2D/D2H/D2D、非默认 Stream 上的 async copy、Stream/Device synchronize、重复分配释放和全部可见 device。CUDA-enabled 构建中的 CPU Runtime 测试仍需通过。
- **当前状态**：未完成

### T4-04 适配并验收第二款平台 Runtime

- **目标**：让 T4-03 的同一逻辑 Runtime 在平台 B 工作；仅对实际 SDK、头文件、编译器或 Runtime API 差异增加最小兼容层，不复制上层模型或算子逻辑。
- **依赖**：T4-03。
- **可能涉及的文件**：
  - `xmake/nvidia.lua`
  - `src/device/nvidia/nvidia_runtime_api.cu`
  - `src/device/nvidia/nvidia_resource.cu`
  - 仅当 T4-01 已确认必要时的厂商兼容头或构建文件
- **验收方法**：在平台 B 独立执行与 T4-03 完全相同的严格 Runtime 测试，设备数必须大于 0，全部同步/异步、Stream、Host/Device 内存和设备切换用例通过；保留平台 B 的独立构建与运行日志，不能引用平台 A 结果代替。
- **当前状态**：未完成

### T4-05 修正 Context、Runtime、Resource 与 Tensor 的设备生命周期

- **目标**：保证每线程、每设备只有一个惰性 Runtime；设备激活先于 Stream/设备资源创建；切换后新 Runtime 被持久保存，存储和资源始终在所属设备上释放；GPU Tensor 能正确创建、加载、读回和调试。
- **依赖**：T4-04。
- **可能涉及的文件**：
  - `src/core/context/context.hpp`
  - `src/core/context/context.cpp`
  - `src/core/runtime/runtime.hpp`
  - `src/core/runtime/runtime.cpp`
  - 必要时 `src/core/storage/storage.cpp`
  - 必要时 `src/tensor/tensor.cpp`
  - `src/device/nvidia/nvidia_resource.cu`
  - `src/device/nvidia/nvidia_resource.cuh`
- **验收方法**：在两平台分别验证同线程 CPU↔GPU、GPU 0↔GPU 1（有多卡时）和重复切换；Tensor 创建、H2D/D2H、view/permute/slice 元数据与内容、debug、销毁均正确；多线程各自建立和销毁 Context 不串设备、不崩溃、无明显泄漏。重新运行 CPU Runtime/Tensor 官方测试和 NVIDIA Runtime 严格测试。
- **当前状态**：未完成

### T4-06 实现 CUDA Add

- **目标**：实现连续同 shape Tensor 的 CUDA Add，支持 F32、F16、BF16，并保证 `out == a` 或 `out == b` 的原地残差写安全。
- **依赖**：T4-05。
- **可能涉及的文件**：
  - `src/ops/add/op.cpp`
  - 新建 `src/ops/add/nvidia/add_nvidia.cu`
  - 新建 `src/ops/add/nvidia/add_nvidia.cuh`
  - `xmake/nvidia.lua`
- **验收方法**：两平台分别通过 `python test/ops/add.py --device nvidia`；额外严格比较三种 dtype 的两类原地别名场景；运行 `python test/ops/add.py --device cpu` 回归。
- **当前状态**：未完成

### T4-07 实现 CUDA Embedding

- **目标**：按 I64 index 从二维权重精确 gather 行，支持 F32、F16、BF16，保持现有 shape/device/dtype 校验。
- **依赖**：T4-06。
- **可能涉及的文件**：
  - `src/ops/embedding/op.cpp`
  - 新建 `src/ops/embedding/nvidia/embedding_nvidia.cu`
  - 新建 `src/ops/embedding/nvidia/embedding_nvidia.cuh`
  - `xmake/nvidia.lua`
- **验收方法**：两平台分别运行 `python test/ops/embedding.py --device nvidia`，但不依赖其缺失的 assert；用独立严格断言覆盖首行、末行、重复 index 和三种 dtype，结果与 reference 完全一致；CPU Embedding 官方脚本回归。
- **当前状态**：未完成

### T4-08 实现 CUDA SwiGLU

- **目标**：实现三种 dtype 的 CUDA SwiGLU，数值路径满足现有容差，并支持模型使用的 `out == gate` 原地调用。
- **依赖**：T4-07。
- **可能涉及的文件**：
  - `src/ops/swiglu/op.cpp`
  - 新建 `src/ops/swiglu/nvidia/swiglu_nvidia.cu`
  - 新建 `src/ops/swiglu/nvidia/swiglu_nvidia.cuh`
  - `xmake/nvidia.lua`
- **验收方法**：两平台分别通过 `python test/ops/swiglu.py --device nvidia`；补充 `out == gate`、小 shape 和 Qwen2 intermediate size 代表 shape 的严格 reference 对照；CPU SwiGLU 回归。
- **当前状态**：未完成

### T4-09 实现 CUDA RMSNorm

- **目标**：沿最后一维完成稳定归约和归一化，支持 F32、F16、BF16，并覆盖 prefill 多行与 decode 单行。
- **依赖**：T4-08。
- **可能涉及的文件**：
  - `src/ops/rms_norm/op.cpp`
  - 新建 `src/ops/rms_norm/nvidia/rms_norm_nvidia.cu`
  - 新建 `src/ops/rms_norm/nvidia/rms_norm_nvidia.cuh`
  - `xmake/nvidia.lua`
- **验收方法**：两平台分别通过 `python test/ops/rms_norm.py --device nvidia`；额外覆盖 Qwen2 的 hidden size 1536、`eps=1e-6`、M=1 BF16 decode，用 FP32 reference 检查误差；CPU RMSNorm 回归。
- **当前状态**：未完成

### T4-10 实现 CUDA RoPE

- **目标**：实现三种 dtype 的位置旋转，正确处理 I64 position id、Q/K 不同 head 数和非零历史位置，并支持 `out == in` 原地调用。
- **依赖**：T4-09。
- **可能涉及的文件**：
  - `src/ops/rope/op.cpp`
  - 新建 `src/ops/rope/nvidia/rope_nvidia.cu`
  - 新建 `src/ops/rope/nvidia/rope_nvidia.cuh`
  - `xmake/nvidia.lua`
- **验收方法**：两平台分别通过 `python test/ops/rope.py --device nvidia`；额外覆盖原地写、较大非零 position、`nh=12`/`nkvh=2`、`dh=128` 和三种 dtype；CPU RoPE 回归。
- **当前状态**：未完成

### T4-11 实现 CUDA Argmax

- **目标**：对一维输入同时返回正确最大值和 I64 首个最大索引，支持 F32、F16、BF16，不能利用官方测试的 `or` 条件只实现一半结果。
- **依赖**：T4-10。
- **可能涉及的文件**：
  - `src/ops/argmax/op.cpp`
  - 新建 `src/ops/argmax/nvidia/argmax_nvidia.cu`
  - 新建 `src/ops/argmax/nvidia/argmax_nvidia.cuh`
  - `xmake/nvidia.lua`
- **验收方法**：两平台分别运行 `python test/ops/argmax.py --device nvidia`；另用严格断言要求 value 与 index 同时正确，覆盖唯一最大值、并列最大值首索引、长 BF16 logits；CPU Argmax 回归。
- **当前状态**：未完成

### T4-12 实现 CUDA Linear 与设备 BLAS 资源

- **目标**：实现 `Y = XW^T + b` 的三种 dtype CUDA 路径，支持可选 bias、M=1 decode 和大矩阵；若使用设备 BLAS，由每设备 Runtime/Resource 持有 handle 并绑定当前 Stream，不在每次调用重复创建。
- **依赖**：T4-11。
- **可能涉及的文件**：
  - `src/ops/linear/op.cpp`
  - 新建 `src/ops/linear/nvidia/linear_nvidia.cu`
  - 新建 `src/ops/linear/nvidia/linear_nvidia.cuh`
  - `src/device/nvidia/nvidia_resource.cu`
  - `src/device/nvidia/nvidia_resource.cuh`
  - 必要时 `src/core/runtime/runtime.hpp`
  - `xmake/nvidia.lua`
- **验收方法**：两平台分别通过 `python test/ops/linear.py --device nvidia` 的全部三 dtype 和大矩阵；补充 `bias=None`、M=1、Qwen2 代表维度和重复调用/切设备验证；确认 BLAS handle 使用正确 Stream 且生命周期无泄漏；CPU Linear 官方脚本及无 bias 补充验证回归。
- **当前状态**：未完成

### T4-13 实现 CUDA GQA Causal Self-Attention

- **目标**：正确实现 GQA、causal offset、prefill 和 `qlen=1, kvlen>1` 增量 decode，支持 F32、F16、BF16，并服从当前 Runtime Stream 顺序。
- **依赖**：T4-12。
- **可能涉及的文件**：
  - `src/ops/self_attention/op.cpp`
  - 新建 `src/ops/self_attention/nvidia/self_attention_nvidia.cu`
  - 新建 `src/ops/self_attention/nvidia/self_attention_nvidia.cuh`
  - 必要时 `src/device/nvidia/nvidia_resource.cu`
  - `xmake/nvidia.lua`
- **验收方法**：两平台最终均须通过 `python test/ops/self_attention.py --device nvidia`，并以独立严格 GPU reference 覆盖 `qlen=kvlen`、`qlen=1/kvlen>1`、`nh=12/nkvh=2`、causal offset 和三种 dtype。若官方脚本先在 reference mask 的设备放置处失败，保留失败证据，不得跳过或放宽断言；仅在官方确认后最小修正测试夹具，后端仍以严格补充测试验收。CPU Self-Attention 回归。
- **当前状态**：未完成

### T4-14 完成双平台 CUDA 算子总回归与范围闭合

- **目标**：确认 8 个正式 CUDA 算子全部进入构建和派发、没有残留占位分支，并落实 T4-01 对 `rearrange` 的范围结论。
- **依赖**：T4-13。
- **可能涉及的文件**：原则上不新增算法修改；检查 `xmake/nvidia.lua`、`src/ops/*/op.cpp` 和所有 `src/ops/*/nvidia/`。若官方确认 `rearrange` 必需，必须先新增独立子任务，不能在本阶段临时捎带实现。
- **验收方法**：在两平台逐个运行实际存在的 Add、Argmax、Embedding、Linear、RMSNorm、RoPE、Self-Attention、SwiGLU NVIDIA 脚本以及 T4-06～T4-13 的严格补充用例；逐个运行对应 CPU 脚本；静态检查 8 个 NVIDIA 派发不再包含占位实现。可用 `--profile` 记录性能，但官方没有固定加速比，正确性是硬门槛。
- **当前状态**：未完成

### T4-15 支持 Qwen2 CUDA 创建、权重加载与生命周期

- **目标**：解除 C API 和 Python 的 CPU-only 限制，继续保持单设备模型；在请求的 CUDA 设备上创建全部权重，复用现有严格 Safetensors 映射，通过同步 H2D 加载并安全 reset/destroy。
- **依赖**：T4-14。
- **可能涉及的文件**：
  - `src/llaisys/qwen2.cc`
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
  - `python/llaisys/models/qwen2.py`
- **验收方法**：在两平台用小型 synthetic 模型验证 create、全部权重映射、reset、重复 destroy；真实目标模型严格加载 `3 + 12 * nlayer = 339` 个 BF16 权重，所有后端 Tensor 位于请求设备，抽样 D2H 原始字节与 Safetensors 源一致；未知/缺失/shape/dtype/错误设备仍明确失败；CPU 权重加载和生命周期回归。此阶段不开始完整 CUDA forward。
- **当前状态**：未完成

### T4-16 实现设备安全的 Qwen2 CUDA Prefill

- **目标**：组合已验收 CUDA 算子完成单设备 prefill；首次 K/V 写入使用设备拷贝，最终 argmax 通过 D2H 返回，所有操作遵守 Runtime Stream，不直接 `std::memcpy` 或解引用设备指针。
- **依赖**：T4-15。
- **可能涉及的文件**：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
  - 必要时 `src/core/runtime/runtime.hpp`
- **验收方法**：两平台的小模型单 token、多 token prefill 分别与 CPU/PyTorch reference 对照 embedding 到最终 argmax，首个分叉点可定位；真实目标模型运行 `--max_steps 1`，完整 token 列表一致；CPU `--max_steps 1` 回归。此阶段只验收首次 prefill，不提前完成增量 decode。
- **当前状态**：未完成

### T4-17 实现设备安全的 KV Cache 增量 Decode 与 Reset

- **目标**：缓存扩容用 D2D 保留有效前缀，新 K/V 写入设备缓存，后续每次只处理一个新 token；reset 清零逻辑长度并复用容量，不按 `maxseq` 一次性分配。
- **依赖**：T4-16。
- **可能涉及的文件**：
  - `src/models/qwen2/model.hpp`
  - `src/models/qwen2/model.cpp`
  - 必要时 `src/core/runtime/runtime.cpp`
- **验收方法**：两平台验证 3-token prefill 后 capacity 惰性增长、跨扩容前缀保持、连续 decode 的 cache length 每步加 1、多 token decode 明确拒绝、reset 后相同 prompt 可复现；小模型每步结果与 CPU 全量重算/PyTorch 一致；真实模型至少通过 `--max_steps 4`，并记录缓存显存未按最大 131072 预分配。CPU KV Cache/generate 回归。
- **当前状态**：未完成

### T4-18 双平台分阶段端到端一致性验收

- **目标**：在两款真实平台上从短生成逐步扩大到正式长度，确认完整 token 列表严格一致，并记录性能和显存。
- **依赖**：T4-17。
- **可能涉及的文件**：原则上不修改官方测试；只根据首个实际分叉点最小修复对应已实现模块。
- **验收方法**：在平台 A、B 分别依次执行：

  ```bash
  python test/test_infer.py --model /path/to/model --test --device nvidia --max_steps 1
  python test/test_infer.py --model /path/to/model --test --device nvidia --max_steps 4
  python test/test_infer.py --model /path/to/model --test --device nvidia --max_steps 128
  python test/test_infer.py --model /path/to/model --test --device nvidia
  ```

  四档均要求完整 token 列表逐项一致；确认长档实际使用增量 KV Cache；分别记录设备/驱动/SDK、构建产物、耗时和峰值显存。若第二平台的 PyTorch 发行版设备映射不同，必须使用 T4-01 已确认的等价严格 reference 流程，不能硬编码期望 token 或只报告理论兼容。
- **当前状态**：未完成

### T4-19 完整回归、CI 与双平台交付检查

- **目标**：确认 Task 4 未破坏 CPU/Task 3，关闭 CUDA 时仍可移植构建，没有修改测试来放宽要求、硬编码输出或复制两套模型逻辑，并形成可复现交付材料。
- **依赖**：T4-18。
- **可能涉及的文件**：原则上不再新增功能修改；必要时 `.github/workflows/build.yaml`、`PROGRESS.md` 或独立交付报告 Markdown。
- **验收方法**：
  - 从干净配置完成 CUDA-off Linux/Windows 构建安装，运行 CPU Runtime、Tensor、8 个算子及 Task 3 正式推理。
  - 在两平台从干净配置完成 CUDA-on 构建，运行严格 Runtime、GPU Tensor、8 个 CUDA 算子及正式 Qwen2 推理。
  - 检查现有官方测试未被删除或放宽，补充测试不固定输入输出 token；运行 `git diff --check` 并审计全部 Task 4 改动。
  - 现有 GitHub Actions 没有 GPU runner，不能用 CPU CI 代替双平台状态；应取得实际平台日志或受控 GPU CI，并如实区分 CPU CI、平台 A、平台 B 的结果。
  - 报告完整复现命令、token 一致性、耗时、峰值显存及逐平台支持状态；未真实运行的项目不得标记通过。
- **当前状态**：未完成

## 当前已知 Task 4 风险与测试缺口

1. 第二款平台已由当前环境的沐曦 MetaX C500 实机证据确定，T4-01 的双平台冻结已完成；但平台 B 的 CUDA 兼容构建（xmake + mxcc/cu-bridge）、工具链与目标模型部署仍需在 T4-02/T4-15 完成并分别在两平台独立验收，不能把单一平台编译成功等同于满足双平台要求。
2. CUDA build 的构建阻塞项（缺失 `xmake/nvidia.lua`、`memcpyAsync` ABI 不匹配）已在 T4-02 全部闭合；剩余阻塞来自 Runtime 算法与 Context 生命周期，构建、Runtime、Context 仍须按顺序独立闭环。
3. 现有官方 Runtime 脚本在零设备时仍会报告通过，不能作为唯一验收；必须增加设备数非零、Stream、async、Host 分配释放和多设备严格检查。
4. Embedding、Argmax 以及模型使用的原地/无 bias 路径存在已知测试缺口；Task 4 必须增加更严格验证，不得利用弱断言。
5. `test/ops/self_attention.py` 的 causal mask 创建未显式放在 query 设备上，GPU reference 可能先于后端失败；必须保留根因证据并使用严格 reference，不得通过跳过测试处理。
6. F16/BF16 kernel、设备 BLAS API 和累加精度可能在不同厂商工具链上存在差异；每个算子都要在两平台分别做 dtype 与数值验收，不能只在最终模型阶段补兼容。
7. 当前 GitHub Actions 只有 CPU runner；最终双平台通过状态必须来自真实设备日志或 GPU CI，报告中应明确区分“未验证”“构建通过”和“运行通过”。

## 当前总状态

- Task 4 需求分析与任务拆分：**已完成**
- Task 4 代码实现：**进行中（构建层已完成，Runtime/算子算法未开始）**
- 双平台选择：**平台 A 已冻结为 node4 的双 NVIDIA A800 80GB；平台 B 已冻结为当前环境的沐曦 MetaX C500（MACA 3.5.3.20 + cu-bridge，CUDA kernel 实机运行验证）**
- 已完成阶段：T4-01、T4-02（平台 B 双开关实机验证；平台 A 构建路径已实现、实机验收待设备权限）。
- 下一个阶段：T4-03 实现 Nvidia CUDA Runtime API（需 node4 同等级设备权限做平台 A 实机验收）。
