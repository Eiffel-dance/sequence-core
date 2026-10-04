# Sequence Core

A dependency-free Python reference implementation for machine-learning, sequence-model, numerical.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现一个仅使用 Python 标准库的 CPU 序列计算核心，提供线性层、逐步前向、梯度累积清零和截断反向传播的最小公开接口，并支持可反向传播的逐步前向会话（start_stream/step/finish_stream），逐行到达的输入得到与批量轨迹一致的输出和梯度。序列输入支持任意宽度的多特征行：Linear 接受至少两个权重时，前 d 个权重作为输入特征系数、最后一个作为循环隐状态系数，反向按输入形状返回每步 d 个输入梯度的行（d 为一时保持标量序列的既有输出类型与梯度列表形状）。截断边界必须真正生效，片段边界的隐状态按接口规则处理，参数更新与公开公式一致；实现不依赖 GPU、网络或第三方框架，数值结果可以通过小规模有限差分脚本复核。TanhSequence 还提供 checkpoint/restore 两个公开入口：checkpoint 在不修改任何状态的前提下返回仅属于当前实例的独立检查点（完整保存隐状态、已产生输出、流会话、截断配置、carry_hidden 规则以及所包装 Linear 的参数、累积梯度和成功前向缓存，且与可变数据不共享引用、可多次恢复）；restore 先校验检查点（损坏、来自其他实例或线性层宽度不兼容时抛出 ValueError 且状态不变），再原子替换当前状态并返回 None，使暂停后的继续计算与从未中断的前向轨迹逐值一致。在此之上，TanhSequence 还提供可跨同宽度实例迁移的 export_state/import_state：export_state 在不修改任何状态的前提下返回仅由字典、列表、有限数字、布尔值、字符串和 None 组成的独立状态对象（带固定整数版本，当前为 1；完整保存隐状态、输出、批量缓存或流会话、初始隐状态、截断与 carry_hidden 规则、有序边界列表、前向参数快照以及 Linear 的权重、偏置、累积梯度和最近前向缓存，无缓存或无会话以 None 表示，且与实例及其他导出结果不共享可变引用）；import_state 接受该对象、其深拷贝或 JSON 往返后的等价值，先做完整的结构与语义校验（类型或字段结构错误、版本或宽度不兼容、非有限数字、轨迹长度或边界不一致、流会话与批量缓存并存、隐状态关系违反截断规则、Linear 缓存形状错误均抛出 ValueError 且目标状态完全不变），再一次性提交并返回 None，使另一个相同宽度的 TanhSequence 从导出时刻继续 step、finish_stream 或任一 backward 入口时与源实例逐值一致，重复导入同一对象结果稳定。
