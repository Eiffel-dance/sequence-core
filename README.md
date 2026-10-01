# Sequence Core

A dependency-free Python reference implementation for machine-learning, sequence-model, numerical.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现一个仅使用 Python 标准库的 CPU 序列计算核心，提供线性层、逐步前向、梯度累积清零和截断反向传播的最小公开接口。截断边界必须真正生效，片段边界的隐状态按接口规则处理，参数更新与公开公式一致；实现不依赖 GPU、网络或第三方框架，数值结果可以通过小规模有限差分脚本复核。
