# 逐点运动传播与可选二维修正

状态：实现及合成开发验证已完成，尚未运行这套新策略的 bush 实验。
当前已接受的结果保持原样。已有 surface、legacy 配置仍可读取和运行；
新实验通过 `configs/neural_pointwise.json` 显式选择新策略。

## 训练与传播（completed_modes v12）

1. 仍使用全前景几何图。每个含 **至少 6 个 Gaussian** 的连通分量有独立学习资格，
   包括原来的大分量；不再要求它至少有两个控制点。
2. 只在这些合格分量中采样控制点。网络保持宽度 256、每控制点 16 维可学习特征，
   图连接、覆盖半径、插值、网络层数、loss 和优化器的其余设置继承准备包。
3. 1–5 点分量的每个点单独选择最近的合格宿主分量，再取该宿主内最多 4 个最近
   Gaussian，使用归一化距离平方倒数作为固定权重。距离相同按原始点编号排序；
   存在零距离时只在零距离邻居间等分。没有距离、表面间隙或宿主歧义的拒绝阈值。
4. 每频率重新组合观测支持。宿主或初始附着点有有效观测贡献时，宿主可参与该频率
   学习。无支持的合格分量逐点回退到有支持的宿主，不使用自身未受监督的网络输出。
   如果该频率不存在任何有支持的运动来源，训练明确报错。
5. 每次网络前向先计算宿主 Gaussian 的最终复数位移，再传播：

   \[
   \Phi_i^{\mathrm{prop}}=\sum_{j\in\mathcal N_i}\beta_{ij}\Phi_j.
   \]

   不额外外推局部旋转；传播点之间不递归传递。传播发生在完整前景渲染和二维 loss
   **之前**，因此这些点的像素残差也会回传给宿主网络。小点没有独立网络输入、
   可学习特征或控制点。形变正则只作用于独立学习区域的原图边，不对逐点传播的小
   分量强制整体刚性。

取消距离门槛保证有宿主时能够取得运动，但不保证空间最近的宿主就是实际相连的枝条。
本版遵照当前前景已经重新划分的假设，没有增加背景识别或补桥规则。

保存 `p_source_mask[K,G]`、`p_neighbor_index/weight[K,G,Q]` 及逐频率运动来源。
支持、频率选择、恢复与缓存身份包含这些内容。v12 的网络、插值、传播可重建导出的
`phi`；旧产物语义不变。

## 可选二维修正（completed_modes v13）

使用 `--refine-observations` 显式启用，默认关闭。它是训练后的独立阶段，固定网络、
宿主运动、静态 Gaussian、相机、复数 alpha 及原归一化尺度。

只允许修正传播点，并且要求其在至少一个可辨识视角中：有实际渲染贡献、中心位于
画面内且深度为正、与缓存表面深度一致、缓存 alpha 至少为 0.05。深度容差沿用
准备包对应旧 graph 的固定端点容差；遮挡或无证据的视角不提供可观测方向。

将有效视角的投影 Jacobian 叠加，计算稳定可观测子空间的投影矩阵 \(P_i\)。
忽略相对奇异值低于 `observable_rtol=0.001` 的弱方向。优化：

\[
\Phi_i=\Phi_i^{\mathrm{prop}}+s_kP_i z_i,\qquad
L=L_{\mathrm{modal}}+lambda\frac1{|T|}\sum_{i\in T}\|P_i z_i\|^2.
\]

这里 \(s_k\) 是训练时冻结的幅度尺度，\(T\) 是允许修正的传播点。单视角不能约束
的方向保留邻域传播值；多个有效视角联合确定可观测方向。有限 prior 表示观测只需
近似匹配，不强制逐点精确满足含噪声的二维目标。

`L_modal` 沿用完整前景特征渲染、复数径向 Huber、固定 alpha、视角能量归一化和
像素 alpha 权重。先合成一个像素的全部贡献，再与该像素目标比较；没有把同一个
像素的目标分别强加给多个 Gaussian。可观测性决定允许改变的变量方向，前向渲染
仍保留所有可用视角的完整物理贡献。

修正配置在 `configs/neural_observation_refinement.json`，默认：

| 参数 | 默认 | 作用 |
|---|---:|---|
| `max_iterations` | 20 | 联合修正迭代上限 |
| `prior_weight` | 0.1 | 抑制偏离邻域传播运动 |
| `initial_step` | 1 | 每次回溯的初始步长 |
| `max_backtracking` | 16 | 单次更新最多回溯次数 |
| `relative_tolerance` | 1e-6 | 总目标相对改善停止阈值 |
| `observable_rtol` | 0.001 | 弱可观测方向阈值 |

采用投影梯度与 Armijo 回溯，只接受目标下降的更新。每完成一个频率保存检查点，
恢复时核对父产物、准备包、配置及修正代码版本。无需重新训练、计算 DFT 或拟合
modal coordinates。真实 bush 的修正耗时及默认权重效果尚待实验评估。

v13 只保存位移增量、可观测方向、修正支持和更新后的采样预测，并引用不可变的
v12 父产物与准备包。加载时合成完整 `phi`。不要删除这些被引用的输入。
宿主逐位不变，零空间校验失败或来源不一致时拒绝加载。Viewer 使用烘焙后的位移，
保留角色和图显示；黄色包含经过二维修正的传播点，原直接观测支持另行保留。

## 同次训练比较开关

在项目根目录的 Anaconda Prompt 中，例如：

```bat
conda activate modal-gaussian
python -m modal_gaussians.cli motion iterate-neural --prepared outputs\bush_neural_dense_controls_001\prepared --config configs\neural_pointwise.json --frequency-hz 0.744 --output outputs\bush_neural_pointwise_0744_001\experiment --stage preview
```

在同一实验上增加修正，不重新训练：

```bat
python -m modal_gaussians.cli motion iterate-neural --prepared outputs\bush_neural_dense_controls_001\prepared --config configs\neural_pointwise.json --frequency-hz 0.744 --output outputs\bush_neural_pointwise_0744_001\experiment --stage preview --refine-observations --refinement-config configs\neural_observation_refinement.json
```

第一条生成 `experiment/preview`；第二条生成
`experiment/refinements/<配置与父结果身份>/preview`。命令末尾打印对应目录，目录中的
`outputs.json` 给出产物路径。原预览不覆盖，改变修正参数也不使网络训练缓存失效。
`--stage` 默认 `modes`；显式 `preview` 才准备 Viser 显示数据。两条命令均不启动
Viewer、不计算 coordinates、不输出 PNG。

## 已完成的开发验证

17 项针对性合成检查通过：大小边界与单控制点资格、逐点宿主和无距离上限、
逐频率支持与回退、传播梯度和模型恢复、v12 保存加载、缓存分层、
v8/v10/v11 加载兼容、共享像素修正及复数符号、宿主和零空间保留、
来源篡改拒绝、开关复用训练及预览路径分离。
其中包含实际 32×32 CUDA 场景的修正保存/恢复，以及选定第二频率的
两步网络训练中断/恢复/导出。未运行真实植物实验，也未检查可视化效果。
