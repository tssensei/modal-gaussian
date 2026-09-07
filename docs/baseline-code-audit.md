# 新 baseline 核心与 legacy 代码审计

> Historical audit before cleanup; see [cleanup record](code-cleanup.md) and [current code map](../src/modal_gaussians/motion/README.md).

审计日期：2026-09-07。代码基准：`1086d7a`（`main`）。

本次仅检查源代码、实际入口、导入关系、配置分派和产物加载逻辑；没有训练、加载大型实验产物、检查可视化或删除代码。下述调用关系为静态代码追踪，不是运行时性能采样。

## 1. 清理所依据的 baseline

当前选中结果为 `outputs/bush_neural_capacity_0744_001/features32`，bush 0.744 Hz；详见 [BASELINE.md](../BASELINE.md)。

- completed modes v16，`neural_component_field_with_stable_donors`。
- 宽度256、局部特征32、消息层3；控制覆盖半径 `0.015L`，预算32768。
- 全前景 mutual 8-NN，最大距离0.008，`graph_edge_filter=none`。
- 至少10个 Gaussian、至少2个控制点、有当前频率有效观测的分量使用自身控制场。
- 可靠传播来源限于至少101个 Gaussian、至少2个控制点且有足够可靠成员的分量。
- 独立控制场先合成，再做固定逐点传播，然后计算完整前景二维模态损失。
- 保留局部形变与旋转变化正则，权重0.1/0.1；没有 observable residual/projector 或观测后修正。
- 默认到3D modes；`preview` 增加二维对比和手动振动显示数据；不默认拟合 modal coordinates。

**当前 baseline 是准备包默认值与 `configs/neural_component_field.json` 合并后的完整配置，不等于裸 CLI 或 dataclass 默认值。** 该 JSON 仍是部分覆盖；例如 `graph_edge_filter` 从准备包继承。清理时要区分“新运行的 baseline 预设”和“旧产物缺字段时的解释”，不能通过修改一个类默认值同时处理二者。

## 2. 实际调用链

```text
cli.py: motion iterate-neural
  -> iteration.iterate_neural
     -> prepared.load_prepared + resolve_config + source/frequency contract
     -> iteration._run_stages
        -> neural_modes.build_neural_modes_artifact
           -> _load_sources (包括旧 graph 和 rigid alignment 适配)
           -> PreparedNeuralInputs.training_inputs
              -> FrozenModalProjector (固定观测/Jacobian/alpha)
              -> geometry_graph.build_geometry_graph_arrays / cache
              -> training_fragments.build_training_controls (策略分派)
                 -> component_field.build_component_controls
                    -> training_fragments.host_subgraph
                    -> geometry_graph.build_control_graph
                    -> pointwise_attachments.assign_points
              -> component_field.observation_inputs
                 -> guarded_attachments.observation_inputs
           -> neural_modes._field_geometry (v16 u_* 分支)
           -> neural_field.train_single_frequency
              -> PerFrequencyModalGNN
              -> model_field / compose_field (包含可微传播)
              -> FrozenModalProjector / modal loss
              -> structural_losses / Adam
           -> 保存最佳网络、phi、图、插值、角色与来源
        -> stage=modes: 返回
        -> stage=preview: rendered_design -> preview
        -> stage=full: 另行调用 coordinates / result / spectrum
```

`component_field.observation_inputs` 在控制分配之前准备来源可见性；图与控制缓存命中时可以复用对应计算。上图按职责展示依赖，未展开检查点、缓存命中和恢复分支。

缓存命中后、训练发布后和预览加载时，严格 loader 仍可能执行图/插值重建、来源核对和保存网络的 `phi` 重现。因此“训练不执行某个旧策略”和“可以删除其文件”不能画等号。

## 3. A 类：新 baseline 核心，保留并作为主阅读入口

以下路径相对于 `src/modal_gaussians/`。

| 文件 | 当前核心职责 | 清理边界 |
|---|---|---|
| `motion/neural/component_field.py` | v16资格判断、整分量自身控制场、可靠 donor、逐点接收关系、角色与诊断 | 最接近当前科学方案的入口；保留其算法，拆掉向旧策略模块借用的通用函数 |
| `motion/neural/geometry_graph.py` | 细图、分量、图距离控制采样、控制图、Wendland稀疏插值 | `none` 建图与控制逻辑是主线；深度筛边策略可隔离，但数组契约和深度阈值输入适配仍有用途 |
| `motion/neural/neural_field.py` | 可学习局部特征、GNN、位移合成、传播、观测/结构损失、Adam与早停 | 主线核心，但包含 v14 残差字段和旧网络配置兼容；不能整文件视为纯v16 |
| `motion/neural/neural_modes.py` | 观测算子、逐频训练组织、数据装配、导出、加载与重现 | 核心与多版本分派混合最严重；后续优先拆策略装配/格式适配，保留当前训练与渲染逻辑 |
| `motion/neural/prepared.py` | 固定观测准备包、图与控制缓存、相机/归一化数据复用 | 保留；旧策略导入、旧结果导入及缓存 revision 需隔离 |
| `motion/neural/iteration.py` | 配置覆盖、选频、缓存/恢复、modes/preview阶段编排 | 保留主线；旧策略、后处理、full评估分支应成为独立可选入口 |
| `motion/neural/preview.py` | 绑定scene、phi、rendered-design和准备包，不制造coordinates | 当前可选预览接口，保留 |

核心函数定位（本次审计版本）：

- `component_field.py:69`：`build_component_controls`。
- `geometry_graph.py:361` / `:480`：`build_geometry_graph_arrays` / `build_control_graph`。
- `neural_field.py:244` / `:302` / `:348` / `:519`：网络、场合成、结构损失、训练。
- `neural_modes.py:293` / `:345` / `:1145`：投影算子、张量装配、训练入口。
- `prepared.py:38` / `:173` / `:197`：准备包、加载、准备入口。
- `iteration.py:22` / `:79` / `:137`：配置、迭代、阶段执行。

## 4. B 类：仍被主线使用的共用模块，不是 legacy

| 文件/模块 | 为什么仍需要 |
|---|---|
| `motion/common/projection.py` | 投影Jacobian、观测采样、特征渲染约定；包含当前支持的相机畸变处理 |
| `motion/common/geometry_ops.py` | 几何采样工具；旧精度约定也应保留到兼容层拆开以后 |
| `motion/common/mode_mapping.py` | 保存顺序、原来源频率与Viewer排序的正确映射 |
| `motion/common/completed_modes.py` | 统一产物类型与版本分派；当前v16和旧结果都通过这里加载 |
| `motion/common/sources.py` | 场景/观测来源契约、固定alpha适配；目前仍依赖旧graph和rigid loader |
| `iteration_cache.py`、`numpy_io.py`、`progress.py` | 缓存、原子发布、身份、保存与进度基础设施 |
| `static.py`、`camera_geometry.py`、`camera_rendering.py` | 冻结场景、相机和实际Gaussian渲染实现 |
| `static_partition.py` | 定义运动前景；不属于优化循环，但当前运动输入依赖其重新划分结果 |
| `rendered_design.py` | 预览二维重建所需，也服务完整评估；不能因为coordinates退出默认流程就删除 |
| `vis/viewer.py`、`vis/spectrum.py` | 手动播放、控制点/角色/几何图、二维对比及可选频谱；保留并分离旧方法adapter |

`topology.py`、`measurements.py`、`modes.py`、`preparation.py`、`synchronization.py` 等属于输入/对齐边界，不能因不在GNN前向中就判成死代码。本轮不清理光流、FFT、3DGS或mask工具。

## 5. C 类：文件源于 legacy，但新 baseline 仍直接复用

这是删除前最需要处理的一组。

| 当前文件与函数 | v16用途 | 可归档部分 | 建议先提取到（尚未实现） |
|---|---|---|---|
| `training_fragments.py:29 host_subgraph` | 构建合格分量的诱导子图 | 原v10整碎片附件/插值算法 | `motion/common/graph_ops.py` |
| `training_fragments.py` 的 `config_class`、`artifact_contract`、`array_names`、`build_training_controls`、roles/validation/diagnostics分派 | 当前v16仍经过这些函数 | v8/10/11/12/14策略分支 | 主线直接装配v16；旧版本交给兼容分派 |
| `pointwise_attachments.py:42 nearest`、`:63 assign_points` | 确定最近donor分量、同分量内最多4点反距离插值、处理并列距离和零距离 | v12 `PointwiseAttachmentConfig`、`build_pointwise_controls`和旧角色规则 | `motion/common/point_transfer.py` |
| `guarded_attachments.py:58 observation_inputs` | donor的中心可见性、深度、alpha检查 | v14弱方向投影、neighbor residual、旧资格条件 | `motion/common/visibility.py` |

当前 `component_field.observation_inputs` 调用整个 guarded helper，再只保留 `h_surface_visible`。该 helper 同时计算 Jacobian；将来可以拆出纯可见性函数，但必须保持已有采样、深度容差和畸变语义，不能顺便换算法。

以下是可直接核对的依赖边：

```text
component_field -> training_fragments.host_subgraph
component_field -> pointwise_attachments.assign_points -> nearest
component_field.observation_inputs -> guarded_attachments.observation_inputs
neural_modes/prepared/iteration -> training_fragments (配置、构建、校验、版本分派)
```

## 6. D 类：退出 baseline 的算法与历史格式

“可归档”指从主线科学逻辑中隔离，不表示现在删文件不会破坏导入或旧结果。

### 6.1 旧 rigid 管线

| 文件 | 历史职责 | 当前保留原因 |
|---|---|---|
| `motion/rigid/structure_graph.py` | topology图、颜色/深度删边、小分量剪枝 | 冷输入仍加载其artifact、身份和几何阈值；旧Viewer也用它 |
| `motion/rigid/rigid.py` | 复数刚体解、trust筛选、刚体产物 | `load_fixed_alignment`仍通过它读取和验证alpha；历史刚体结果需加载 |
| `motion/rigid/motion_fill.py` | v1/v2单视角提升、固定anchor、LSMR顺序补全 | 旧completed modes兼容与历史CLI |
| `motion/rigid/motion_basis.py` | v3共享权重、候选、距离先验、FISTA | v3及v4–v7共同的数组/场重建和校验工具 |
| `motion/rigid/motion_basis_frequency.py` | v4/v6逐频权重、逐频可信集合 | 旧逐频结果及v5/v7的父产物 |
| `motion/rigid/motion_basis_green.py` | v5/v7固定蓝点的绿色refinement | 保留的非神经参考结果就是v7 |

新训练不使用刚体运动或trust，但目前还不能移除 `rigid.py`、`structure_graph.py` 的加载接口。需要先把alpha、可辨识视角、来源以及donor使用的几何容差变成明确的中性输入适配。已有 `rigid/` 已经独立，第一轮不必再搬它来增加路径变更。

### 6.2 旧 neural 策略

| 版本 | 方法 | 主要位置 | 对v16的关系 |
|---|---|---|---|
| v8 | 原始神经复数场 | `neural_modes.py`、`neural_field.py` 的无training-fill路径 | 基础网络/场数学仍共用；旧装配和格式保留在compat |
| v9 | 训练后整碎片传播 | `fragment_propagation.py` | 算法不执行；默认配置、导入、hash与旧baseline加载仍引用此模块 |
| v10 | 训练内整碎片控制插值 | `training_fragments.py` | 算法已退出；通用子图函数和分派仍在主线 |
| v11 | 表面附件 | `surface_attachments.py` | 算法已退出；仍被裸入口默认值、无条件import和hash引用 |
| v12 | 早期逐点传播与小分量学习规则 | `pointwise_attachments.py` | 提取 `nearest/assign_points` 后再隔离旧策略 |
| v13 | v12观测后修正 | `observation_refinement.py` | v16明确拒绝启用；旧loader/CLI仍需要 |
| v14 | 稳定邻域残差、可观测方向投影 | `guarded_attachments.py`、`neural_field.py` 的residual分支 | 只复用visibility helper，旧残差表示不在baseline |
| v15 | guarded观测后修正 | `observation_refinement.py` | 不在baseline |
| v16 | 整分量控制场、独立donor条件 | `component_field.py` | 当前主线；同为v16的旧单控制点配置仍需解释兼容 |

`fragment_propagation.py`、`surface_attachments.py`、`observation_refinement.py` 的算法适合整体归档，但要先消除主线的导入、配置与hash耦合。不要简单逐文件删除。

### 6.3 共用核心文件内部的 legacy 分支

- `neural_field.py`：`residual_projector`、`residual_mask`、`residual_prior_weight`及邻域残差组合属于v14路线；`transfer_*`本身是v16必需，不能一并删除。无local_features的旧网络形状也是旧模型加载契约。
- `neural_modes.py::_field_geometry`：v16使用 `u_*` 语义，并复用 `t_interpolation_*` 存储；`p_*`、`h_*`及旧 `t_motion_component_index` 装配需隔离。不能仅按前缀删全部 `t_*`。
- `neural_modes.py`：配置序列化、`_semantics`、数组校验、版本发布、来源重放、prefix导出混在训练组织中。建议提取策略/格式adapter，而不是复制整个文件给每个版本。
- `geometry_graph.py`：三态深度边证据、unknown短边策略不是当前训练选项；`depth_thresholds_from_manifest`仍给donor准备提供阈值，不随深度筛边分支一起删除。
- `iteration.py`：v8后传播、观测refinement与`full`分支都不在当前modes/preview路径。它们仍有CLI和旧结果用途。
- `vis/viewer.py`：旧sequential/basis角色、rigid trust着色是legacy adapter；相机、手动时钟、模式排序、当前角色和控制点开关是共享功能。

## 7. E 类：可选下游能力，不应等同于无用旧代码

`direct_coordinates.py`、`physics_coordinates.py`、`result.py` 和相关CLI不参与默认3D模式优化。它们仍服务显式完整评估、历史完整结果的加载、部分准备包的 `--from-result` 导入。

建议从主线入口改为延迟导入/可选完整评估模块；本次不建议直接删除。尤其 `iteration.py` 顶层导入 direct coordinates，`_run_stages` 无条件导入 result，尚未实现依赖隔离。

## 8. 配置与文档归类

| 文件 | 分类 |
|---|---|
| `configs/neural_component_field.json` | 当前baseline覆盖项，保留；下一步将新运行preset与旧配置解码分开 |
| `configs/neural_local_features.json` | 旧局部特征实验配置，不是当前32维component-field baseline配置 |
| `configs/neural_guarded.json` | legacy v14 |
| `configs/neural_observation_refinement.json` | legacy v13/v15 |
| `configs/neural_pointwise.json` | legacy v12 |
| `configs/neural_surface_attachments.json` | legacy v11 |
| `docs/component-field.md`、`BASELINE.md` | 当前方法/已选结果入口 |
| `docs/guarded-motion-fill.md`、`docs/pointwise-motion-fill.md`、`docs/neural_surface_attachments*.md` | 历史方案与验证记录，可归档 |
| `README.md`、`src/modal_gaussians/motion/README.md` | 含多个阶段说明；需要区分当前入口与历史复现说明，不能据旧说明选择baseline参数 |
| `skills/modal-gaussians-pipeline/` | 继续保留；后续同步当前v16 preset和训练内传播语义，不应删除运行恢复指导 |

## 9. 清理前必须识别的耦合

### 9.1 文件名和内容参与缓存revision

`iteration_cache.module_revision` 对模块名及其源文件hash计算身份。

- `iteration.py` 的基础 `neural_revision` 包含 `training_fragments`、`fragment_propagation`、`surface_attachments`，v16再加入 `component_field`、`guarded_attachments`、`pointwise_attachments`。
- `prepared.py` 的控制缓存也有同样的旧模块revision依赖。
- `iteration.json`严格比较配置与code contract；文件移动、重命名、甚至仅改变被hash文件的注释都可能改变缓存键或令旧实验目录拒绝resume。

不能把“数值没变”直接等同于“可以继续用原目录resume”。清理后的新运行应使用明确的新revision/缓存命名空间；旧结果身份、旧配置与路径保持原样，不能伪造旧hash来获取命中。若以后需要恢复旧工作目录，应保留相应旧代码执行方式或设计显式迁移。

### 9.2 legacy loader有重建依赖

`load_neural_completed_modes` 不只是读取 `phi`：它还校验图/插值与来源，并用保存网络重现位移。旧basis loader也会重建图、候选或混合位移。

因此兼容层需要保存重建/解码逻辑，不只是保留一个manifest读取函数。Viewer帧播放使用烘焙 `phi`，但初始化经过这些loader。

### 9.3 裸默认值不是baseline

`NeuralModesConfig`、`NeuralFieldConfig`、`fit-neural`仍保留宽度64、local feature 0等早期默认；无显式fragment配置的部分入口选择surface策略。`ComponentFieldConfig.from_dict`则为旧v16缺失 `min_learning_controls` 的配置保留值1。

建议新运行通过单一baseline preset解析；旧artifact通过版本化解码得到历史缺省值。不能把旧配置缺省值统一改成当前32维或2控制点，否则会破坏身份和模型shape。

### 9.4 “不执行旧求解”不等于“不读旧输入”

即使使用准备包，当前 `build_neural_modes_artifact` 仍先调用 `_load_sources`；固定alignment通过rigid loader验证，旧observed graph仍在来源契约中，depth阈值也仍用于donor可见性。

彻底断开旧rigid生产端依赖，需要中性alignment/visibility输入契约和旧artifact adapter；不能直接删 `--graph` 或 `--alignment-from`。

## 10. 建议清理顺序（未执行）

1. **先抽共用函数**：`host_subgraph`、`nearest/assign_points`、donor可见性；使v16科学模块不再向旧策略实现借函数。
2. **明确当前preset与策略接口**：新运行显式使用component-field；配置解析、控制构建、role、数组契约分别按策略/版本dispatch。保留旧缺字段解释。
3. **隔离neural旧策略**：将v9–v15生产与格式逻辑放入明确的legacy区域；v8基础数学复用当前核心，不机械复制。先保留 `rigid/` 已有目录边界。
4. **拆重型混合模块**：优先处理 `neural_modes.py` 的格式/策略adapter和 `neural_field.py` 的guarded残差分支，保留固定场数学与网络state_dict键。随后整理Viewer方法adapter。
5. **隔离来源与可选评估**：中性固定alpha/visibility契约；full/coordinates懒加载；保留旧完整结果和新preview入口。
6. **处理revision与旧结果**：新代码使用新contract；旧结果通过兼容loader读取。同步文档和skill，最后才移除没有引用且不再需要复现的代码。

首轮可以采用以下边界，文件名为建议，不代表已经建立：

```text
motion/
  common/             # 中性输入、投影、频率映射、子图、可见性、逐点转移
  neural/             # component-field、图/控制、GNN、场/损失、训练、准备和预览
  legacy/neural/      # 原fragment/surface/pointwise/guarded/refinement策略
  rigid/              # 已有旧刚体目录，先保留
  common/completed_modes.py  # 稳定入口，按版本懒加载adapter
```

清理应保留当前bush 32维baseline、现有corn结果和原非神经参考，不删除输出或共享缓存。变更验证只做受影响的少量检查：公共函数结果一致、当前v16合成场/传播与梯度、代表旧格式加载/身份、baseline配置解析、CLI/preview导入与频率映射。不要把本次代码整理变成新的真实训练或自动可视化评估。

## 11. 本次结论

最有价值的整理对象是 **`neural/` 内混合的新旧策略、重复的策略分派以及过宽的缓存依赖**。当前没有证据支持直接删除整个旧策略文件后仍能保持主线和旧结果可用。应先提取共用能力并隔离兼容边界，再决定哪些历史生产入口真正停止维护。
