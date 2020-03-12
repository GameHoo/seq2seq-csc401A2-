使用神经网络把英语翻译为法语

- 计算都设计成批量计算，方便并行计算加速。

- 使用seq2seq模型

- 加入Attention机制

- 不同的energy_score计算方法(cosine, additive , dot-product, scaled-dot-product)

- 用 beam_search 选出k个候选项，用来测试性能。

  