# 【第八周作业】

1. 使用中文对联数据集训练带有attention的seq2seq模型，利用tensorboard跟踪。
https://www.kaggle.com/datasets/jiaminggogogo/chinese-couplets
2. 尝试encoder hidden state不同的返回形式（concat和add）
3. 编写并实现seq2seq attention版的推理实现。

## 作业流程

1. **数据准备**  
    - 下载中文对联数据集：[Chinese Couplets Dataset](https://www.kaggle.com/datasets/jiaminggogogo/chinese-couplets)  
    - 对数据进行预处理，分词并构建词汇表。

2. **模型构建**  
    - 搭建带有Attention机制的Seq2Seq模型。  
    - 使用不同的Encoder hidden state返回形式（如concat和add）进行实验。

3. **模型训练**  
    - 使用预处理后的数据集训练模型。  
    - 利用TensorBoard跟踪训练过程，包括损失值、注意力权重等。

4. **推理实现**  
    - 编写推理代码，输入上联生成下联。  
    - 可视化Attention权重，分析模型生成结果。

5. **结果分析**  
    - 比较不同Encoder hidden state返回形式对模型性能的影响。  
    - 总结实验结果，撰写实验报告。