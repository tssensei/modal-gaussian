# Baseline 代码清理记录

日期：2026-09-07。基于 `1086d7a` 后的 baseline 审计实施。

## 当前入口

当前科学方法仍为 v16 `neural_component_field_with_stable_donors`。网络、控制点采样、插值、传播、损失与精度不变；选中的实验结果也不变。

阅读入口见 [motion 代码地图](../src/modal_gaussians/motion/README.md)。

| 层次 | 位置与职责 |
|---|---|
| 新运行配置 | `motion/neural/baseline.py`：256 宽度、32 维特征、3 层及当前 component-field 策略 |
| 当前科学逻辑 | `geometry_graph.py` → `component_field.py` → `neural_field.py` |
| 训练与阶段组织 | `neural_modes.py`、`prepared.py`、`iteration.py` |
| 格式校验与重放 | `neural/artifacts.py`；统一入口仍为 `common/completed_modes.py` |
| 共用能力 | `common/graph_ops.py`、`point_transfer.py`、`visibility.py` |
| 旧神经策略 | `legacy/neural/`：fragment、surface、pointwise、guarded、observation refinement |
| 旧刚体管线 | 继续保留在 `rigid/`，兼容历史结果和固定 alpha 来源 |

六个旧神经模块已实际迁移，原目录不留重复副本。子图、最近邻传播和可见性实现已提取；重复的策略分派也收拢到 `neural/strategies.py`。历史数组字段在轻量 schema 中共享，避免仅为读取格式声明就导入旧求解器。

## 配置边界

- `fit-neural`、显式来源的新准备包、未提供 `--config` 的 `iterate-neural` 使用当前 preset。
- 显式 iteration 配置继续覆盖准备包的记录值；推荐从 [neural_component_field.json](../configs/neural_component_field.json) 修改参数。该文件现在明确记录图连接参数，避免意外继承旧深度筛边设置。
- 不修改历史 dataclass 缺省值：旧网络缺少特征维度时仍解释为 0，旧 v16 缺少最少控制点字段时仍按原规则加载。
- 默认仍到 `modes`；`preview` 增加二维对比及手动振动输入；只有明确的 `full` 才导入和执行坐标拟合。

## 减少的读写与检查

1. 神经产物在临时目录通过一次完整校验后发布，直接返回该次校验得到的数组，不再从最终目录重读、重建一次。
2. 准备包同样复用发布前已加载的对象，取消发布后的第二次加载。
3. 图和控制缓存保留写入校验，成功发布后直接返回已核对的数据，取消随后的重复解压和哈希。并发任务发布的结果仍独立验证，并清理本次竞争失败的临时目录。
4. rendered-design 可以接收已验证的 completed modes；预览发布可以接收已验证的准备包、模式和 design。继续核对路径与来源绑定，不为这些同一阶段的对象重新走完整 loader。
5. 训练和控制缓存的代码依赖按策略选择。修改隔离的历史策略实现不会使 v16 缓存失效。
6. 未变化的源代码哈希在当前进程内复用；数据文件不使用这种基于文件属性的快捷验证。

独立加载产物时仍保留内容哈希、来源、数组合法性、固定几何与网络重现检查。原子发布和恢复检查点也保留。这里减少的是重复执行，不是取消科学输入校验。

## 保留与兼容

旧格式仍经兼容 loader 加载，数组字段、网络 state dict 和产物身份算法保持原含义。当前训练还需要旧 graph 和 rigid alignment 的来源校验，不能删除相应 loader。共享场计算中的少量旧格式适配也继续保留，避免复制整套数学实现。

新 iteration 契约为 version 2，代码 revision 随模块拆分变化。新实验需要使用新目录；不要将旧 revision 的实验目录当作本次代码的 resume。已有结果和 Viewer 命令不需要迁移。引用已迁移 Python 模块的旧脚本需要改用 `motion.legacy.neural` 路径。

## 开发验证

相关合成检查覆盖控制场、传播及梯度、模态与结构损失、历史格式加载、v16 保存/重放、频率索引、CLI 配置、缓存并发/损坏/失效、预览身份绑定和按需导入。迁移引起的测试路径与旧默认值断言已同步，并定向重跑通过；合计覆盖 127 项测试。

没有运行真实训练、读取大型实验产物、拟合 modal coordinates、启动或检查 Viewer、生成 PNG。没有修改或清理已有输出和 baseline。本次没有实测端到端提速比例。
