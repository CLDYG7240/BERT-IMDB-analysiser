# IMDB影评情感分析模型

本项目是一个基于BERT（Bidirectional Encoder Representations from Transformers）模型的情感分析项目，旨在对IMDB影评进行正面或负面情感的分类。

## 模型介绍

我们使用了预训练的BERT模型（`bert-base-uncased`版本），并在IMDB影评数据集上进行了微调，以适应情感分析任务。BERT是一种基于transformer的模型，它通过双向训练从文本中学习到深层次的语言表示。

## 环境要求

- Python 3.6+
- PyTorch 1.7.1+
- Transformers 4.5.1+
- Datasets 1.6.2+

## 安装指南

首先克隆仓库到本地：

```bash
git clone [项目仓库地址]
```

然后安装所需的依赖：

```bash
pip install -r requirements.txt
```

## 训练模型

要训练模型，请运行以下命令：

```bash
python train.py
```

这将在IMDB数据集上开始训练过程，并在`./results`文件夹中保存模型。

## 使用模型进行预测

一旦模型训练完成，你可以使用它来对新的影评文本进行情感预测。使用以下代码进行预测：

```python
from predict import predict_sentiment

print(predict_sentiment("This movie was absolutely wonderful!"))
```

## 贡献

如果你想为此项目贡献代码，欢迎提交Pull Request或开Issue讨论你的想法。

## 许可

此项目遵循MIT许可证。有关详细信息，请参阅`LICENSE`文件。# -BERT-
