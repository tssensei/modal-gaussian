**Modal Gaussians：CPU/GPU 优化机会检查（2026-09-19）**

更新（2026-09-20）：本文以下内容保留为优化前检查记录。GNN 固定量预计算、
GPU best-state、合并 loss 读回及共享主干单次反传已经实现；软传播已默认使用
CuPy float64。alpha 现已实现 GPU 几何分解与缓存、局部求解、残差、两点数值求导
以及完整有界 TRF，新实验默认 GPU。批次按 alpha → 软传播 → GNN 分阶段使用 GPU，
CPU 建图可与准备重叠。详见 [GPU alpha](gpu-alpha.md)。随后经用户授权，Bush 0.25 Hz
单次对比测得 CPU 11.24 秒、GPU 磁盘缓存命中 8.90 秒（1.26×）、几何常驻 6.96 秒
（1.62×）；GPU 首次构建缓存 58.85 秒，反而更慢。这是 alpha 阶段单频率测速，
不是全管线加速比。未恢复停止的 Bush 批次。

固定模态投影已有可选的 `neural.modal_projection_backend="cached"` 路径：
在训练进程内复用 gsplat 投影、图像块排序和径向采样网格，保留原像素混合及其
特征反传；默认 `dynamic` 仍走原路径。尚未实现完整稀疏投影矩阵。随后经用户授权，
Bush 三视角各后端 60 次短基准测得投影/图像损失及运动场反传为 57.66 → 12.02 ms
（4.80×），初始化缓存 0.099 秒，额外常驻显存 56.9 MiB。未运行完整训练；
GNN、结构正则和优化器不在该计时内。详见 [投影测速](modal-projection-benchmark-20260920.md)。

本次检查当前工作树的执行路径和已有日志；没有训练、重算缓存、运行真实数据 validation 或改动计算逻辑。范围是现有静态场景之后的 SEA-RAFT、共享 FFT、选频/导出、逐频率准备、软权图、控制点传播、GNN、结果发布，并附带检查可选的时间系数/RGB 拟合。COLMAP 和静态 3DGS 重建不在本轮已测量范围内。

这里区分代码中确定存在的重复工作、已有阶段计时，以及尚需测量的加速假设。没有微观 profiler 数据，不能把整个阶段耗时归因于其中某一行，也不能承诺加速倍数。

**已有证据**

Bush 批次记录目录：[current_5000](C:/Users/zitengsong/Documents/school/research/modal-gaussian/scene_library/bush/experiments/uniform60/current_5000)。这些是历史并发、续训记录，不是独占机器的统一 benchmark。

| 阶段 | 记录数量 | 中位墙钟时间 | 解释 |
|---|---:|---:|---|
| 逐频率软传播，缓存未命中 | 26 | 614.04 秒 | CPU 准备阶段的主要开销 |
| 跨视角复数幅相对齐 | 29 | 179.87 秒 | 各频率变化大，22.36–800.01 秒 |
| network_optimization | 31 | 455.24 秒 | 包含续训，不能当作从零训练 5000 步耗时 |
| modal_similarity_graph | 29 | 3.63 秒 | 计算软边权本身当前不慢 |
| selected_modal_publish | 29 | 3.66 秒 | 独立准备产物写出 |
| shared_control_geometry，CPU 命中 | 29 | 0.11 秒 | 已经共享控制点布局 |
| frequency_control_weights，训练端命中 | 25 | 0.18 秒 | GPU 任务已直接消费 CPU 准备的缓存 |

不同阶段和频率会重叠；嵌套计时不能相加。缓存命中样本与未命中样本不是同一项工作的重复测量。

共享 FFT 的 manifest 记录：Bush 三视角共 1025.14 秒，Corn 两视角共 240.97 秒。Bush greedy60 报告记录 173.64 秒，uniform60 约 0.012 秒。这些是一次性产物，不能算进每个新频率的训练开销。

当前设备读到 RTX 5090，显存 32607 MiB，CPU 有 32 个逻辑处理器。历史 gpu_usage.csv 的 1204 个样本：GPU utilization 均值 31.34%、中位数 29%、P90 55%、最大 85%；显存中位数 7996 MiB、最大 9455 MiB。这是混合 CPU 准备和 GPU 训练期间的采样，既不能代表纯训练利用率，也不能据此线性推算还能加多少 GPU 任务。CSV 的 active_processes 是全部活动阶段数，包含 CPU 任务。

**从输入到结果的机会**

| 环节 | 当前实现 | 优先尝试 | 资源方向 / 优先级 |
|---|---|---|---|
| SEA-RAFT | 单图对推理；读图、GPU 推理、回传、压缩写盘依次进行 | 有界图像预取和写盘队列；再尝试真正的小批量推理 | CPU/GPU 重叠；新场景时中优先级 |
| 共享 FFT | 视角和空间 tile 顺序处理；NumPy FFT；同步 Zarr 写入 | tile/shard 对齐、读算写重叠，之后再比较多线程 FFT 或 GPU FFT | CPU/I/O 优先；一次性工作 |
| 选频/导出 | greedy 已有批量 Schur 求解；导出直接读缓存 | 复用 sufficient statistics；分块读像素和导出预取 | 低优先级 |
| 参考几何/KNN | 已复用；首次构建仍有 Python 循环 | 新场景时并行 KDTree 查询，保留确定性 tie-breaking | CPU；当前批次低优先级 |
| 幅相对齐 | 逐 Gaussian SVD；重复小矩阵特征分解和有限差分残差调用 | 缓存固定几何部分；按块批量求解；之后考虑 GPU | CPU 向量化/共享优先；高优先级 |
| 逐频率软边权 | NumPy 分块，逐视角取端点证据 | 缓存投影位置、可见性等几何量 | CPU；低优先级 |
| 固定控制图 | 布局、几何支持域已缓存共享 | 保持现有复用 | 已完成的优化 |
| 软传播 | 每频率进程池；每控制点自适应重复 Dijkstra | 减少重复搜索，统一并发预算，再考虑共享内存 | CPU；高优先级 |
| GNN | GPU；每步仍重算静态量、多次主干反传和 CPU 同步 | 预计算、合并同步、GPU 保存 best state | GPU；先做的小改动 |
| 固定模态投影 | 每视角每步调用完整 rasterizer | 复用固定投影/排序，进一步考虑精确稀疏投影算子 | GPU；潜力较大、工作量较大 |
| 发布/加载 | 独立 NPZ 仍包含重复静态数组 | 后续按共享几何 + 频率增量组织；复用已加载对象 | I/O/内存；低优先级 |
| 时间系数/RGB（可选） | 已有分块/部分缓存，但有重复解码和数据扫描 | 图像预取、缓存参考光流、充分统计量 | CPU/GPU 重叠；不属于 modes_ready 必需阶段 |

**1. GNN：先减少每步重复工作，改动最集中**

代码：[neural_field.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/motion/neural/neural_field.py:238)。

- weighted_neighbor_mean 每层重新拼双向边、计算固定归一化分母。可在每个频率初始化时保存双向边和归一化权重；不同频率权重不能混用。
- compose_field 每步重新算控制点到 Gaussian 的固定 offset，donor 目标的 torch.unique 也每步重算。可预计算，换取少量常驻显存。
- structural_losses 每步重新筛固定边、计算长度/方向和权重和。可在该频率的 geometry 初始化时完成。
- baseline 的 rotation_weight=0，但 control rotation loss 仍被计算。可跳过该损失分支；这不等于关掉局部旋转场，也不能删掉 Gaussian strain 中使用的旋转项。日志应区分“未计算”与真实测得的零。
- 每个视角 float(term.detach())、各项损失读回和 best-state 更新都会触发主机交互。把需要的损失与状态合成一次读回，并保持每步早停判断和有限值保护。
- 每次 best loss 改善都执行 _cpu_snapshot(model.state_dict())。best state 可先在 GPU detach().clone()，仍按原 checkpoint 节奏导出到 CPU。不能保存会随训练变化的 state_dict 引用；best 与 latest、优化器和 RNG 的续训语义都要保留。

PyTorch 官方建议减少 .item()/CPU 拷贝等同步点；这一建议与当前代码的具体调用位置吻合。[官方性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)

另一个中等工作量机会在 [objective](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/motion/neural/neural_field.py:605)：每个视角调用 backward(retain_graph=True)，之后正则再 backward，共享 GNN/插值会重复反传。当前逐视角处理是为了控制 rasterizer 内存，不宜简单同时保留全部视角图。可以逐视角累积对最终 Gaussian field 的梯度，再向共享 GNN 反传一次；正则还依赖 rotation/control fields，必须保留这些梯度路径和复数梯度约定。这保持数学目标，但浮点累加顺序可能变化。

**2. 软传播：最大的重复 CPU 准备项**

代码：[control_propagation.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/motion/neural/control_propagation.py:25)、[shared_controls.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/motion/neural/shared_controls.py:86)。

当前确实已经并行：每频率一个 spawn 进程池，16 个 control 为一块。每个 control 调用 SciPy Dijkstra，若未覆盖所有原支持点，就扩大 limit 并从头重算。弱边为 0.05 时，最大路径拉伸为 20，最坏会经历约 1、2、4、8、16、20 倍的多次完整重启。每次求解还产生全图距离向量，最后只取该 control 的支持点。

最值得研究的是一次搜索持续推进，直到目标支持点全部出队，以避免重新搜索和无关输出。这里“目标出队”指最短距离已经确定，不能在第一次发现目标时就停止。要保留允许路径绕出原支持区域的语义；限制成支持域内部子图不是等价加速。

先保留成熟的 SciPy 内核，增加很小的阶段计时/重试计数来辨别主要开销，再决定是否需要原生 target-aware 实现。用 Python heap 重写 Dijkstra 可能更慢；直接使用最大 limit 也可能多搜索大片区域。二者都没有现成加速保证。

内存方面，Windows spawn 让每个子进程持有 CSR/support 副本；频率之间拓扑可共享、边代价不同。只有确认副本/启动成为瓶颈时再加 mmap/shared memory。持久进程池也不应抢在搜索本身的优化之前。

GPU 最短路可作为后续原型，但并非最稳的第一步：这里是稀疏、不规则、多个源、局部目标集合。不要一次生成全部 control × Gaussian 的距离；以约 3428 × 231761 的 float64 距离表计算，仅该表就约 5.9 GiB，还不含队列、中间结果和训练。

**3. 跨视角 alpha：可以继续共享的几何计算很多**

代码：[synchronization.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/synchronization.py:231)。

_build_constraints 按 Gaussian 做带权投影矩阵的 SVD，建立左零空间约束；_profiled_residuals 每次根据当前 alpha 形成很多 3×3 系统，用批量 eigh 消去局部 3D 位移；SciPy least_squares 未传入显式 Jacobian，因此还会为数值差分重复计算残差。

可以跨频率缓存 observation 分组、固定 sqrt(weight)×Jacobian、几何 Gram、以及所需的紧凑分解。缓存键必须包含几何/对应关系/权重/视角子集。不能缓存最终 alpha，也不能复用依赖当前模态观测的 RHS、归一化、Huber scale、可辨识判定。3×3 逆矩阵还依赖求解中的 alpha，不能全部预先固定。

执行顺序建议是先消除逐点 Python 调用和重复几何分解，再考虑解析导数或 GPU 批量残差。现在已有部分批量 NumPy 实现，不能简单称为“全部逐点串行”。把残差放 GPU、外层仍留在 SciPy，会产生反复读回；需要同时衡量传输和 RTX 5090 上该精度的实际吞吐。降低精度可能影响病态视角的排除结果，不属于纯并行改动。

**4. 固定模态渲染：比把所有东西搬上 GPU 更值得研究**

代码：[FrozenModalProjector](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/motion/neural/neural_modes.py:309)、[render_features](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/static.py:547)。

模态训练的 Gaussian 几何、相机、opacity、排序和 alpha 是固定的；变化的是四通道运动特征。但每个优化步骤仍调用完整 rasterization 路径。

数学上，该阶段可写成固定线性投影 p_v = A_v J_v Phi：J_v 是每 Gaussian 的投影 Jacobian，A_v 包含可见渲染贡献、采样及 alpha 归一化。实部/虚部分别应用同一算子。可以先复用投影几何/排序，再研究缓存实际贡献的稀疏 A_v，只做 GPU 稀疏乘法及其转置反传。

这可能同时摊薄多个频率的训练成本，但要先评估实际非零数和显存。现有 topology 的有限 contributor 对应关系不自动等于完整渲染算子；不能直接替代而声称监督完全相同。Corn 手选主体路径的背景遮挡、相机畸变处理也必须保留。当前日志没有把 rasterizer 时间单独拆出，暂不能量化收益。

此优化只适用于固定几何的 modal-image 投影。逐帧 RGB 渲染移动 Gaussian 后，可见性和排序会改变，不能复用同一个固定 A_v。

**5. FFT：先处理数据布局与写入，再决定 CPU/GPU**

代码：[temporal_rfft_tiles](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/flow/spectrum.py:47)、[build_spectrum](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/spectrum_cache.py:176)、[Zarr 创建](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/flow/storage.py:41)。

当前 build 使用 32×64 左右的空间 tile，而典型 Zarr 空间 shard 是 128×128。一个 shard 被多个相邻 tile 分次更新，存在部分块读写/压缩重复的机会。应尝试按 shard 聚合输出、有界预取下一块、由独占该 shard 的 writer 写入；实际压缩/I/O 占比仍需拆分计时，不能把 17 分钟全部叫作 FFT 运算时间。[Zarr 官方说明](https://zarr.readthedocs.io/en/stable/user-guide/performance/#sharding)

FFT 后端方面，先比较已安装 SciPy 的 rfft(workers=...)；它提供内部并行参数，无需新增依赖。GPU FFT 可以作为后一项，必须把读取、传输、FFT、回传和压缩全部计入收益。若和 GNN 同时跑，会争用同一块 GPU。[SciPy rfft](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfft.html)

保持原始序列去均值、对称 Hann、补零、未归一化复数系数、共享频率表和全分辨率背景。更换后端并不保证字节级相同，不能沿用旧身份把新数值覆盖进去。现有完成缓存应继续复用，不因性能优化重算。

**6. SEA-RAFT：真正的流水线和小 batch**

代码：[sea_raft.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/flow/sea_raft.py:107)。

目前 batch_size 用于 CPU 缓冲/Zarr 写入；model 实际收到单帧图对。可先让 CPU 解码下一批、GPU 算当前图对、CPU 压缩上一批，以有界队列限制内存。真正的图对 batching 可以再试，但需保留 reference→每帧方向、位移单位和原分辨率。

上游 SEA-RAFT 的 fnet(image1) 只依赖固定 reference，可以研究缓存；cnet 输入却是 cat([image1,image2])，不能整体缓存 reference context。修改第三方 forward 比预取复杂，所以排在后面。帧结果所有权和写入完成状态要明确，不能复用仍在异步读取的缓冲区。

**7. 建图、选频和产物：低优先级，但有具体机会**

- [geometry_graph.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/motion/neural/geometry_graph.py:338) 的 KDTree 查询 workers=1，候选排序/互近邻用 Python set。首次新场景可加 CPU 查询并行或向量化，但保留重复距离下的确定性排序。控制点 FPS 的下一点依赖之前的覆盖，不能直接任意并行所有采样步骤；独立 component/固定支持搜索更适合并行。目前这些结果已缓存，当前多频率不必重做。
- [modal_similarity.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/motion/neural/modal_similarity.py:138) 可复用视角投影、深度可见性、patch 索引和近距离边候选；模态幅值、可靠性和边决策继续逐频率算。当前 3.6 秒量级，不建议为它先增加 GPU 模块。
- [spectrum_selection.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/spectrum_selection.py:85) 已将 greedy 候选变成批量 Schur residual 与 2×2 求解，不能把它当成尚未优化的旧 greedy。已有 statistics.npz 可作为后续不同选频数量的复用输入；像素读取可按存储块组织。导出已直接读取缓存 bin，不再运行 DFT。
- [selected_modal.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/motion/neural/selected_modal.py:326) 和 neural_modes 发布仍写包含固定几何的独立 NPZ。计算共享已经实现，磁盘/加载共享并不完全。后续可改为引用不可变几何加逐频率增量，但涉及格式兼容；目前发布只占数秒，不宜先做大迁移。
- v16 加载仍需恢复 rotation/control displacement，所以网络重放不全是 validation。聚合 mode bank 已有一次恢复后缓存的路径，优先复用，不把必要重放简单删除。

**8. 调度：已有 CPU/GPU 双队列，下一步管理总预算**

代码：[batch.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/motion/neural/batch.py:215)。

当前 CPU 准备与 GPU 训练已经重叠；可以动态调 CPU/GPU workers 和 propagation_workers，BLAS 线程数也已限制。这些不需要重新实现。

但频率级并行和传播级并行相乘。例如 4 个频率各 6 个传播进程，就是最多 24 个搜索进程，还要加 alpha、压缩和 GPU 主线程的 CPU 工作。不是每个子进程都一直用满 BLAS 线程，因此也不能简单乘线程数当真实负载。

后续建议用全局 CPU 预算，并为 GPU launch、I/O 保留余量；根据 ready_for_gpu 调整准备的提前量，避免无界准备。若要加 GPU 任务，以完成频率/小时及峰值内存为依据，不能只看显存尚有空余。改进监控时分别记录 active_cpu/active_gpu/ready 数和阶段时间，现有 active_processes 字段不足以解释 GPU 利用率。

更激进的一进程多频率 GPU batching 可以摊薄固定几何、调度和 rasterization，但必须保存每频率独立的参数、Adam 状态、早停和软边权；每频率不同 donor/support 也可能引入形状差异。不是将 60 个频率一次堆进去，也不需要 DDP：当前只有一块 GPU，任务之间又是独立模型。

**9. 可选的时间系数与 RGB 路径**

这部分不是主线 modes_ready 的必需阶段，单独列出以免混入默认耗时。

- [rendered_design.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/rendered_design.py:609) 已支持 modes_per_batch 多频率特征渲染。先复用这项能力，不再另做逐 mode 的重复渲染入口。
- [direct_coordinates.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/direct_coordinates.py:435) 已每视角复用一次 Gram/Cholesky；主要机会是每个时间块重复读取 reference flow，以及后续残差统计再次扫描 flow。缓存参考像素，并复用拟合时累积的 AᵀA、Aᵀy、||y||²，可以计算相同目标下的残差统计，避免再读全序列。病态情况下充分统计量公式的消减误差需要考虑。
- [rgb_coordinates.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/rgb_coordinates.py:198) 只缓存 8 个 CPU 帧，随机遍历一千多帧时跨 epoch 命中通常很少。优先有界 PNG 解码预取；需要时按当前 pyramid scale 缓存解码帧，不直接把全部全分辨率 float RGB 搬到 GPU。
- [rgb_fitting.py](C:/Users/zitengsong/Documents/school/research/modal-gaussian/src/modal_gaussians/rgb_fitting.py:183) 的 batch_size 是逐帧渲染/反传后合并 optimizer step，尚不是同时渲染一批帧。真正批量渲染要额外显存，且不能破坏未采样帧不更新 Adam momentum 的现有语义。固定空间 mode 的混合矩阵可以留在 GPU，但动态可见性不可固定。

**建议实施顺序**

1. 先做 GNN 静态量预计算、best-state GPU 保留、合并损失读回、跳过零权重旋转正则计算。这组改动集中，直接减少每一步浪费，不改变模型结构。
2. CPU 准备优先处理 alpha 几何共享与软传播重复搜索。软传播是已有记录中更大的重复开销；alpha 几何共享通常更容易先做出小改动。
3. 梳理逐视角反传，并研究固定 modal projector 的几何/排序复用；再考虑更大范围的精确稀疏算子和多频率 batching。
4. 新场景需要重新生成上游数据时，再做 FFT shard/流水线和 SEA-RAFT 预取。现有缓存保持复用。

首次实施可在原有正常计算中附加轻量阶段计时，不追加真实数据 validation 或完整缓存扫描。需要确认数学等价的改动可用最小合成输入检查；不以降低分辨率、减步数、更改 loss、截断传播路径或换近似图来冒充同一管线的性能提升。
