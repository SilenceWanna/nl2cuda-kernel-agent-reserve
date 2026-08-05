# CASE_EVIDENCE.md — 逐 case 实测例证附录（SKILL.md 的证据支撑，非方法论正文）

> **本文件是什么**：`skill/SKILL.md` 的方法论正文只讲**不点 case 名的通用原理**（瓶颈类型、能否赢判据、
> 边界分类）。本附录收录支撑那些原理的**逐 case 实测例证**——哪个 case 证了哪条规律、用什么手段、
> 达到多少加速比。这些是"证据/案例库"，供人类回顾与信任校准。
>
> **⚠️ 三宿主测试时必须删除本文件**（`prepare_cleanroom.sh` 会自动删）：它逐 case 记录了解法要点 +
> 性能数字，agent 读到 = 开卷考（2026-07-30 gptme cholesky 实测 grep 命中矩阵答案而作废）。SKILL 正文
> 只给方向、不给某 case 的完整公式/性能，故正文保留、本附录剥离。

## 瓶颈类型 → 实测例证（对应 SKILL "瓶颈诊断→策略选择"表）

| 瓶颈类型 | 手段 | 实测例证（case 前/反加速比） |
|---------|------|------------------------------|
| 反向多遍访存 | 多梯度融合到一 kernel、shared 私有累积 + chunk 末 atomicAdd | l2norm_scale 反向 0.84→1.60×；⚠️ per-element global atomicAdd 融合翻车 0.85→0.26× |
| 串行依赖(扫描) | CUB BlockScan/DeviceScan；反向=反向扫描 | scan 前 3.4×/反 4.1× |
| 矩阵乘 | cuBLAS Sgemm 拼 + 手写融合尾；关 TF32 守 fp32 | GEMM+bias+gelu 前 1.19/反 1.16；attention 前 1.3× |
| 变系数递推 | reference log 空间 O(T²) 下三角；candidate 手写 O(T) 递推 | gated_ssm 前 7.8/反 13.8（CUDA 能做稳定 O(T)、PyTorch 不能） |
| 数据依赖写入 | atomicAdd + 选低冲突规模 + float4 gather 反向 | scatter_add S=32768(低冲突) 前 2.19；高冲突 S=4096 仅 1.02× |
| 间接寻址 | 手写融合 CSR kernel（gather+乘加一趟） | spmv 前 5.97/反 2.48 |

## 五类形态 → 归类的 case（对应 SKILL "形态分类 + 能否赢判据"总纲）

- **稳赢区**（访存更少 或 解析结构）：rbf(距离)、cosine_sim(点积 2.12)、softmax_ce(exp规约 1.35)、GroupNorm(大分组reduce 1.28~1.40)、scan/linear_ssm/gated_ssm(cumsum/O(T)递推)、conv1d(4.78/1.86)、spmv(间接寻址 5.97/2.48)、gridsample(数据依赖采样 2.11/2.26)、segment_softmax(变长分段 1.34/2.12)、geglu(逐元素融合链**反向** 2.36)、tridiag(线性求解**反向**=解伴随 1.27/9.80)、cholesky(**反向**=Φ算子 反 1.25)、dynamic_grid_evolution(2D数据依赖stencil+gather反向 4.63/5.26)、integral_image(2D前缀和 前2.89/反1.32；前向靠 ILP 4行流水翻盘——初版单线程串行沿H仅0.81~1.0，改流水后0.99→2.89，是"打不过先别认①本征、多为②实现力"的活教材)。
- **擦线区**（小 reduce）：LayerNorm/RMSNorm(D~1K)/l2norm/welford/temperature_softmax 前向。float4+寄存器缓存或擦 1.0~1.1，多规模易掉，须 3 连 + 计算主导区验。
- **带宽墙区**（纯访存低算术强度前向）：maxpool 前向(读4写1，三宿主 1.03/0.998/1.02)、geglu 前向(逐元素)、小 reduce 归一化前向。反向常仍可赢。
- **厂商库墙区**（前向 baseline 是 NVIDIA 厂商成品）：cholesky 前向(cuSOLVER potrf，手写分块 0.31×)。反向若有解析结构（Φ 算子）仍可能小胜(cholesky 反 1.25)。
- **宿主分层区**（融合密集，考验实现力）：GEMM+bias+gelu、online_softmax、causal_attn——codex 类强宿主能赢、中弱宿主可能挂。

## 边界识别实测例证（对应 SKILL "识别本征边界"）

- **maxpool 前向**：三宿主一致过不了（1.03/0.998/1.02）——带宽墙铁证。反向 4.58× 赢（argmax 散回可优化）。
- **归一化 reduce 家族由规模分野**：
  - 小 reduce（~1K，LayerNorm/RMSNorm D=1024/l2norm/welford/temperature_softmax）**前向**→ 带宽墙。LayerNorm 前向坐实：放大规模掉破 1.05，寄存器缓存省第二遍 X 读无感（1.012→1.019）；codex 交付仓端到端复验前向 1.00×（1.37TB/s 与 torch.compile 重合）。
  - 大 reduce/点积（~万级或 exp）→ 可真赢。GroupNorm(组内 1.2万~4万)前向 1.28~1.40；cosine_sim 2.12；softmax_ce 1.35。
  - **反向要看 kernel 结构，不是一概带宽墙（2026-07-31 端到端复验纠正旧结论）**：LayerNorm 反向若把 dX+dgamma+dbeta **融进单个 kernel、grad_Y/X 各只读一遍**（codex 交付仓版），访存约为"拆分两 kernel（各读一遍=共两遍）"的一半，**能翻过带宽墙真赢**——A100 实测 LN_B=262144 反 1.89×、524288 反 1.94×（换 2× 规模加速比稳且略升，baseline 4.7/9.3ms 绝对耗时合理，非弱 baseline）。**旧 §29"LayerNorm 反向也带宽墙 0.99"是拆分两 kernel 版的结论，被单 kernel 全融合版推翻**。GroupNorm/cosine_sim/softmax_ce 反向亦可优化(1.2~1.4)。**阶段 11.1 复审坐实（2026-08-03，计算主导区实测）——整行归一化家族反向普遍非带宽墙**：welford 反向拆分版仅 1.20×，仿 codex 重构成单 kernel 全融合（一 block 管 32 行、grad/X 各读一遍、行内 dX + 跨行累积 dgamma/dbeta）后 **1.92×（2× 规模 1.94× 稳）**——与 LayerNorm 翻墙同构，证明"拆分→单 kernel 全融合"是**可复制的通用手法**非 codex 偶然；temperature_softmax（单梯度 dscores）反向 1.64×、l2norm_scale（已融合版）反向 1.60× 计算主导区实测本就达标。**结论：小 reduce 归一化家族"前向带宽墙、反向靠单 kernel 全融合可翻墙"是普遍规律**；§37 对 welford/temp_softmax 反向的"暂存疑"已解除。
- **本征边界 vs 没优化够**：前者三宿主一致卡同点且放大仍打不过（maxpool 前向、LayerNorm **前向**）；后者同 case 有宿主赢有宿主挂（attention/scatter——codex 赢 gptme/aider 挂，实现力空间）。⚠️ LayerNorm **反向**不属带宽墙——单 kernel 全融合能真赢（见上），说明"带宽墙"要分前/反向、分 kernel 结构判，别整例算子一刀切。

- **带宽墙实例穷尽验证汇总（2026-08-04，全部计算主导区双规模实测坐实）**：

  | case | 前向 | 反向 | 类别 |
  |------|------|------|------|
  | layernorm | 1.01× 墙 | 1.48→1.69× | 小 reduce 归一化：前墙 / 反融合翻墙 |
  | welford | 1.01× 墙 | 1.20→**1.92×** | 同上（重构融合翻墙） |
  | l2norm_scale | 1.02× 墙 | 1.60× | 同上（已融合） |
  | temperature_softmax | 1.01× 墙 | 1.64× | 同上（单梯度反向也赢） |
  | rmsnorm | 1.0× 墙 | 融合翻墙 / aider 拆分擦线 | 同上（端到端三宿主） |
  | maxpool | 1.02× 墙 | 3.16× | **纯访存前向真本征** |
  | geglu | 1.03× 墙 | 2.41× | **纯访存前向真本征** |
  | softmax_ce | **1.33→1.84× 赢** | 1.26→1.39× 赢 | **exp 规约非带宽墙**（算术强度够，前反向都真赢） |

  **三条铁律坐实**：① 纯访存前向（maxpool 读4写1 / geglu 逐元素）= 真本征带宽墙，融合也救不了、认边界；② 小 reduce 归一化前向 = 本征带宽墙，但**其反向靠单 kernel 全融合可翻墙（1.5~1.9×）**；③ exp 规约 / 大 reduce / 点积（softmax_ce / GroupNorm / cosine_sim）**前向就真赢，根本不是带宽墙**——别把"有规约"误当"带宽墙"，看**算术强度**（~1K 逐元素→墙、exp/万级→赢）。**判定必须计算主导区多规模实测**，短核假象会把三类都误判。

## LayerNorm 反向"能优化≠能赢"实测教训（对应 SKILL "识别本征边界"的告诫）

阶段 8 曾以为"缓存 mean/rstd + 二维分块列规约 + float4"把 LayerNorm 前 1.08/反 1.25"稳过 5%"——但多规模复测推翻：那是 LN_B=32768 短核虚高，放大到计算主导区前反向均掉破 1.05，是带宽墙本征边界（低算术强度整行归一化 D=1024）。**缓存 mean/rstd 消除 O(B·D²) 重算是真修复**（正确性 + 避免灾难，朴素版曾卡 1228ms）、二维分块列规约是正确结构，但**性能达标本身是短核假象**。教训：归一化反向"能优化"（结构上）≠"能赢 torch.compile"（计算主导区）。

## 红线：辅助原语 vs 直调目标算子（对应 SKILL 红线 §1 细分）实测

- **合规**：codex Cholesky 用 cuBLAS TRSM/GEMM 拼分块 Cholesky（TRSM/GEMM 是积木，分块调度自己写）。
- **违规**：aider Cholesky `dlopen` 直调 `cusolverDnSpotrf`（potrf=Cholesky 分解本身=成品）——即便集成低效没赢也不算数。
- **判据**：库调用是**积木**（还需自己拼算法）还是**成品**（直接就是本题答案）——成品级直调禁。
