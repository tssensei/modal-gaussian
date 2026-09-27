**Ambient Gaussian 本地源码核查与公平比较方案 — 2026-09-26**

核查对象为同级目录 `summer2023/x3d/experimental/forestGaussians`，以及
`summer2023/.vscode/launch.json`。历史 HEAD 为 `80c898d`，提交日期
2024-01-31，作者 ShihMengLi，提交说明为 `Update before Siggraph in a rush`。
工作区含有未提交修改；以下区分当前代码、Git 历史和论文描述。

本次读取源码、历史配置和目录清单，执行 CPU 合成检查；没有修改 summer2023、
安装依赖、运行真实场景训练/评估/导出或更改现有 Modal Gaussians 产物。
此报告不代表已复现论文分数，也不判断哪个方法画面更好。

**结论**

- 已有可研究的 Ambient 实现，不能再以“官网未公开代码”作为本地对比的障碍。
  但此 checkout 是开发代码和本地修改的组合，尚不能当作论文评测可直接复现的发行版。
- 核心是位置编码网络预测 DCT 系数，由帧号计算 Gaussian 轨迹。与我们的
  “复数空间振型 + 自由逐帧 q”不同，它原生定义了留出时刻的运动。
- 当前启动配方意图运行 Instant-NGP 深度、80k 静态、30k 动态；但实际优化器
  判断存在阶段时钟问题，按该配置静态阶段最多调用 29,999 次 optimizer.step。
- `abla_images.txt` 才是 COLMAP 路线的损失屏蔽入口。相机列表的 `train/test`
  名称不能证明真正的数据划分；默认日志也不是仅留出帧的指标。
- 当前渲染/指标链路没有完整接通；LPIPS、SSIM、量化方式与我们的 evaluator 不一致。
  正式对比应由同一个目标清单和指标实现评分。
- 本地未找到配方引用的 Forest 输入或完整训练结果，不能确认实际论文划分、
  数据尺寸、最终配置、运行耗时和论文分数。

**版本与实际入口**

README 仍是继承的 4DGS README；`arguments/baseline/default.py` 是另一个基线路线，
不是 Ambient 完整配方。有效线索是当前根目录的
[launch.json](../../summer2023/.vscode/launch.json)、
[train.py](../../summer2023/x3d/experimental/forestGaussians/train.py) 和
[hypernerf/default.py](../../summer2023/x3d/experimental/forestGaussians/arguments/hypernerf/default.py)。
入口以 `python -m x3d.experimental.forestGaussians.train` 组织，工作目录为 summer2023。
本文不提供直接执行配方，因为下述训练/评估问题尚未修复。

本地增量包括 `gaussian_traj` 分支、直接逐 Gaussian DCT 参数的保存/恢复/增删、
`shiftview`、帧号上界修正及保存步数变化。历史留出脚本、部分运行配置、
`full_eval.py` 和顶层许可证文件在工作区已删除，仍可只读查看 Git 历史。
没有将这些删除恢复到工作区。

当前 `gaussian_traj` 默认 False，主训练启动配置未开启它。该分支与网络预测系数
不是同一方法；`scene/gaussian_model.py:455` 附近读取位移系数时使用了
`dctcoef_q` 前缀，不能无核验地用于基线加载。先锁定网络预测 DCT 的路线。

`GSPLAT_DATA`、`GSPLAT_DATA_w_depth`、`GSPLAT_DATA_w_depth_abla` 在该 checkout
不存在；`4D_GSPLAT_OUT` 和 `Pretrained` 均为空。这里的结论仅覆盖这些已检查位置，
不表示用户磁盘其他地方没有数据。

**训练阶段与学习参数**

| 阶段 | 当前启动配方/代码 | 对比较的影响 |
| --- | --- | --- |
| Instant-NGP | 1000 iter/epoch × 10 epochs，图像长边 1000；生成 depth mean/mode | 预处理是该场景上额外学习的步骤，应计时并执行相同划分 |
| 静态 coarse | 配置 80k；前 9k 允许密度控制，之后可学习门控裁剪 | 不是仅 3k 的粗场景；必须先修复实际更新数问题 |
| 静态目标 | L1 + 0.2 DSSIM + 0.05 depth L2 + 0.0005 mask penalty | mask 是每个 Gaussian 的可学习门控，不是前景视频分割监督 |
| 动态 fine | 配置 30k；DCT、deform_scale=0.001、no_ds；no_do 默认 True | 学习位置与四元数轨迹；没有逐帧自由 q 表 |
| 动态目标 | L1 + grid regularization + 4.0 rigidity + 4.0 relative rotation | 此路径的 DSSIM 和 depth loss 只在 coarse 启用 |
| Gaussian 参数 | fine 的 optimizer 仍包含 xyz、全部 SH、opacity、scale、rotation | `no_ds` 只禁止时间相关尺度形变，不代表基准尺度冻结 |
| 数量 | 合并配置 densify_until_iter=45k，fine 预算为 30k | fine 的密度操作并未被整体关闭 |
| 场景边界 | 相机范围归一化、编码查询边界策略 | 没有与我们人工主体分区等价的硬背景零运动合同 |

证据：[训练损失与密度控制](../../summer2023/x3d/experimental/forestGaussians/train.py)，
[优化器参数组](../../summer2023/x3d/experimental/forestGaussians/scene/gaussian_model.py)，
[编码器](../../summer2023/x3d/experimental/forestGaussians/scene/hexplane.py)。

KNN 约束使用基准 Gaussian 位置，邻居权重为 exp(-2000 × 距离平方)，数量变化时重建。
比较当前帧与上一帧的边相对旋转和边位移；没有我们按频率的参考图传播和 donor 关系。
权重数值不能与我们的刚性损失权重直接横比，距离归一化和损失形式不同。

**运动模型：与当前 20 振型的关键差别**

[scene/deformation.py](../../summer2023/x3d/experimental/forestGaussians/scene/deformation.py)
在 `dct_deform=True` 时将网络输入时间置零。空间编码与 MLP 给每个位置预测固定
DCT 系数，真正的时间变化来自
[utils/deform_utils.py](../../summer2023/x3d/experimental/forestGaussians/utils/deform_utils.py)
中的余弦基。

对位移的一维分量，当前实现等价于：

\[
\Delta x_i(j)=s\sqrt{2/K}\sum_{k=1}^{K} a_{ik}
\cos\left(\frac{\pi(2j+1)k}{2N}\right),\qquad s=0.001.
\]

四元数的四个分量另行预测 DCT 系数，加到基准四元数后归一化；没有将同一个
deform_scale 乘到四元数增量上。DC 项关闭，基准位置/旋转承担静态项。
训练帧到测试帧使用同一公式，推理不需要目标 RGB。

当前启动配方 `dct_k=0.25`，代码按最大帧号计算 K 并四舍五入。若文件名是
连续的 0..361，N=362，则 K=90；按真实 30 FPS 解释，最高余弦频率约 3.729 Hz。
这是源码公式的算例，不是已经运行了我们的 sweep。名称不从零开始或有帧号缺口时，
N/K 也会受影响，因此必须保留并核对时间索引。

我们的表达式是 Re(sum(phi_k × q_k(j)))，RGB 精修的 q 是自由逐帧复数变量。
20 个复数空间振型与 90 个固定 DCT 时间基不能按“频率个数”直接对齐：
两者控制的自由度和时间约束不同。公平比较应报告各自原配方、容量/存储/耗时，
另做容量消融，不应擅自把 Ambient K 压到 20 后视为原方法。

**输入、时间和相机合同**

COLMAP 路线要求 `images/` 与 `sparse/0/`，图像文件名主体必须可转换成整数。
`time_index` 来自文件名，`time=frame_index/(最大注册帧号+1)`；没有读取真实 FPS
或多录制 sequence ID。渲染视频固定写 30 FPS。

若有深度监督，读取 `nerf_depth_means/nerf_depths/<name>.png` 的 16-bit 深度和
对应 `nerf_depth_ranges/<name>.txt`，然后进行场景尺度变换。
不能把我们 DA3 深度直接改名当作它的原深度产物；必须核对单位、相机深度定义、
归一化与有效区域，且方法变化要单独记录。

Gaussian 相机加载器接受 SIMPLE_RADIAL/OPENCV 名称，但只使用焦距计算 FoV，
不传递畸变和非居中主点。Instant-NGP 部分则固定读取 `camdata[1]` 的内参。
因此不能直接将我们含畸变/不同相机的原始输入丢进去期待像素对齐；推荐共同生成
经验证的去畸变 pinhole 输入，并做投影一致性检查。

这份实现没有原生异步多视频的独立时间轨迹。把 sweep 和 view1 首尾拼接后赋予
一个全局 DCT 时间轴，会增加人为的跨录制时间约束，不能称为无改动地支持多序列。
若设计多序列适配，应明确标记为适配版，保持录制内真实时钟和独立时间参数。

证据：[dataset_readers.py:166](../../summer2023/x3d/experimental/forestGaussians/scene/dataset_readers.py)，
[NGP 数据加载器](../../summer2023/x3d/experimental/forestGaussians/Instant_NGP/datasets/instant_ngp_dataset.py)，
[Camera](../../summer2023/x3d/experimental/forestGaussians/scene/cameras.py)。

**留出片段的实际机制与限制**

1. `scene/dataset_readers.py:182` 读取 `abla_images.txt`，0 对应 `disable_loss=True`。
   不存在该文件时，所有注册帧均可监督训练。
2. `train.py:263` 将被屏蔽帧的总 loss 乘零；coarse 和 fine 都经过该判断。
   帧仍被采样、渲染并推进步数，Adam 动量及密度统计/日程也未整体跳过。
   因此它不是“只在训练帧列表中采样”的实现，步数与有效监督次数要分开记录。
3. Instant-NGP 的 ray sampling 也读取同名 mask。该处用布尔列表直接索引
   COLMAP 迭代顺序，没有显式按文件名对齐；有注册缺帧或顺序变化时需先检查映射。
4. `train_cam_infos=cam_infos; test_cam_infos=train_cam_infos`，两者是同一列表。
   默认 training_report 对两组使用同样的抽样索引，不按 disable_loss 筛选。
   因而日志里的 test PSNR 不是单独的留出段指标。
5. COLMAP 相机和点云直接加载已有文件；abla mask 不会重新构建其来源。
   严格留出协议必须另行定义允许的相机标定信息和点云/颜色来源。

历史生成脚本可通过以下只读命令查看：

```powershell
git -C ..\summer2023 show 80c898d:x3d/experimental/forestGaussians/Instant_NGP/gen_abla_mask.py
```

其示例给 `dyntree_IMG_1698_gaochen` 留出 `[20,50]`、`[120,150]`、`[220,250]`，
并同时包含区间两端，完整连续序列上合计 **93 帧**。
[论文 §4.5](https://arxiv.org/html/2406.09395v1) 描述为 3×30=90 帧。
该脚本只是历史示例，不能据此断言论文实际跑错、其他场景都用相同索引，
或认为已经找到了最终论文划分。必须取得实际 mask/数据/结果配置进一步确认。

**必须先处理的训练与评测问题**

| 项目 | 已核实的源码行为 | 对复现的影响 |
| --- | --- | --- |
| 阶段步数 | train.py:349 用 `iteration < opt.iterations` 控制两个阶段的 optimizer.step | 当前 coarse=80k/fine=30k 配置，静态最多 29,999 次参数更新，后续仍可裁剪 |
| 导出入口 | render_set 必需 9 个位置参数；train/test 两个调用只给 7 个 | 不跳过这两个分支会参数绑定失败；video 分支参数数目完整 |
| 评分输入 | render.py 的逐帧 renders/gt PNG 保存被注释，主要写 MP4 | metrics.py 所需 test/*/renders 和 gt 不由当前配方完整生成 |
| 评分子集 | metrics.py 遍历 renders 中所有文件，不读取 abla mask | 即使补回 PNG，仍必须明确只计算所选目标帧 |
| 错误报告 | metrics.py 用宽泛 except 打印失败 | 命令结束不能作为评测成功证据，需检查逐帧数量与结果文件 |
| 参数合并 | get_combined_args 对已有 cfg_args 字段不采用命令行覆盖 | 修改 -s 等参数后仍可能使用旧绑定；适配时必须打印并检查生效配置 |
| 恢复状态 | checkpoint 保存位于当前更新之前；恢复跳过到该步，未保存完整随机/采样状态 | 不能未经检查宣称是严格恢复同一训练轨迹 |
| 无深度路径 | train.py:200 无条件访问 viewpoint_cam.depth.cuda() | 仅关闭深度开关不一定构成可用的无深度运行配方 |

这些是针对已检查代码的结论，不是对论文结果真伪或作者最终内部版本的判断。
不能用明显未工作的配置训练一个较弱基线，再将它作为 Ambient 正常质量。

**指标实现不一致**

| 项目 | summer2023 当前 metrics.py/工具 | Modal Gaussians 当前 evaluator |
| --- | --- | --- |
| 输入 | 读取渲染和 GT 图像文件 | 在线 float32 渲染，对应原始 PNG |
| LPIPS | lpipsPyTorch，VGG | lpips，Alex v0.1，spatial=True |
| SSIM | 11×11 Gaussian，sigma=1.5，卷积补零并统计边缘 | pytorch-msssim 有效窗口；掩码路径只统计全有效窗口 |
| 支持区域 | 全图，没有稳定化 valid 支持合同 | 固定视角共用 valid，原生 sweep 全图 |
| PSNR | MSE 转 PSNR，没有显式 floor | MSE floor=1e-12，上限 120 dB |
| 汇总 | 文件逐帧均值 | 帧均值、像素加权 pooled、等视角汇总 |

不能直接拼接两边已有 JSON 到一张表。推荐双方输出同一目标帧/相机下的
无损浮点或共同量化图像，由同一份 evaluator 计算。若要复现历史数字，另保留
明确标注的原指标协议；原指标不能与新协议混排。

文件证据：[metrics.py](../../summer2023/x3d/experimental/forestGaussians/metrics.py)，
[loss_utils.py](../../summer2023/x3d/experimental/forestGaussians/utils/loss_utils.py)，
[我们的 evaluator](../src/modal_gaussians/results/evaluation.py)，
[静态 SSIM 实现](../src/modal_gaussians/geometry/training.py)。

**建议采用的最小对齐顺序**

1. 锁定 summer2023 的具体提交和必要修补，区分本地新增方法分支。修复阶段更新
   和渲染/评测连接，并用小型输入确认实际更新数、输出帧身份和加载前后一致性。
2. 定义一份共享目标清单：源录制、原帧号/时间戳、相机、图像 hash、有效区域、
   train/val/test 与允许的观测。沿用我们已有绑定设施，不另建一套训练框架。
3. 先选一个开发场景确认两种方法可以对相同相机/像素输出，使用统一指标。
   这一步可评价已观测视频拟合，明确不将其标为泛化结果。
4. 要对齐 Ambient 单目留出时间协议，需要相同训练视频和确切留出片段；我们
   必须增加只依赖训练观测的 q 时间推断，同时禁止测试 RGB/flow/FFT/选频进入训练。
   仅有 Forest 单目移动视频时，现有固定视角 modal observation 如何获得也尚待解决。
5. 若研究目标保持“sweep + 异步固定录制”，让双方获得相同观测，明确 Ambient
   的多序列适配。动态新视角质量最好另采同步留出相机作为 GT；现有异步 view1
   不能作为 sweep 同一时刻的真值。
6. 正式实验分别报告各方法完整管线质量和相同总耗时预算质量。共享初始静态场景
   的运动比较可作为额外消融，不能冒充完整原方法。预处理、深度、模态恢复、拟合、
   精修和最终模型的成本分别记录。

我们的当前结果可保留为开发记录。正式划分变化后，凡是已使用测试帧像素的
静态场景、深度学习、FFT/频率选择、振型/q/场景精修产物都需要重新检查来源，
按依赖重建；不能通过重新绑定测试标签使旧产物成为独立评测结果。

**本次实际验证**

在现有 modal-gaussian Python 环境中执行 CPU 检查，仅提取被核查函数或 AST：

- 历史 mask 的闭区间计数为 93，断言通过。
- 提取 train.py 的真实 optimizer.step 条件，在 coarse=80k/fine=30k 上计数为
  29,999，断言通过；没有运行 80k 训练。
- AST 检查 render_set 的必要参数数目与调用点，确认两个 7 参数调用不完整。
- 直接调用原 idct：362 帧示例 K=90，公式误差 2.09e-8 以下，系数梯度有限。
- 同一对 32×32 的人工边界图像，旧 SSIM=0.2071957，有效窗口 SSIM=0.4320422。
  这只证明实现不同；边界占比很大的合成图不能用于推断真实 540p/1080p 差值。
- 解析本地 JSONC 启动配方，确认 80k/30k、DCT=0.25 和上述 loss 参数。
- 输入目录存在性与输出目录内容计数已核实，没有读取或运行真实 Forest 图像。

尚未验证：Windows CUDA 扩展可运行性、完整训练、真实 mask 与最终论文配置、
真实帧指标、视频画面质量、运行速度。仓库中现存的 Linux .so 不能作为 Windows
运行已经就绪的证据；环境兼容性应在获准运行基线时单独检查。
