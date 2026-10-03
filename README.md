# Sequence Core

A dependency-free Python reference implementation for machine-learning, sequence-model, numerical.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现一个仅使用 Python 标准库的 CPU 序列计算核心，提供线性层、逐步前向、梯度累积清零和截断反向传播的最小公开接口，并支持可反向传播的逐步前向会话（start_stream/step/finish_stream），逐行到达的输入得到与批量轨迹一致的输出和梯度。序列输入支持任意宽度的多特征行：Linear 接受至少两个权重时，前 d 个权重作为输入特征系数、最后一个作为循环隐状态系数，反向按输入形状返回每步 d 个输入梯度的行（d 为一时保持标量序列的既有输出类型与梯度列表形状）。截断边界必须真正生效，片段边界的隐状态按接口规则处理，参数更新与公开公式一致；除固定 truncate 网格外，批量 forward 可通过 segment_starts 逐行声明非等距片段边界，流式会话中 step 的 segment_start 标记只影响当前行，两者合并同一套边界规则，finish_stream 记录的缓存与同配置批量调用逐项一致；实现不依赖 GPU、网络或第三方框架，数值结果可以通过小规模有限差分脚本复核。
