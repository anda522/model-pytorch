这是一个个人深度学习模型实现的工具库项目，主要用于学习目的。项目包含了常见深度学习模型的PyTorch实现、深度学习工具库使用示例以及常用代码片段。

主要为个人收集整理，可能出bug。

## 📊 已实现模型

### 计算机视觉
| 模型 | 状态 | 变体 | 特性 | 论文链接 |
|------|------|------|------|----------|
| ResNet | YES | ResNet-18/34/50/101/152 | 残差连接、预训练权重 | [paper](https://arxiv.org/abs/1512.03385) |
| ViT | YES | ViT-Base/16, ViT-Large/16 | 位置编码、patch embedding | [paper](https://arxiv.org/abs/2010.11929) |
| ConvNeXt | NO | Tiny/Small/Base/Large | 现代化ConvNet | [paper](https://arxiv.org/abs/2201.03545) |

### 自然语言处理
| 模型 | 状态 | 变体 | 特性 | 论文链接 |
|------|------|------|------|----------|
| Transformer | YES | Base/Large | Encoder-Decoder、多头注意力 | [paper](https://arxiv.org/abs/1706.03762) |
| BERT | NO | Base/Large | 双向编码器、MLM预训练 | [paper](https://arxiv.org/abs/1810.04805) |
| GPT | YES | GPT-1/2/3风格 | 自回归生成、Decoder-only | [paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| LLaMA | NO | 7B/13B/30B/65B | 高效大模型 | [paper](https://arxiv.org/abs/2302.13971) |

## paper

| 模型 | 状态 | 特性 | 论文链接 |
|------|------|------|----------|
| BYOL | YES | 对比学习方法 | |

⭐ 如果这个项目对你有帮助，请给个星标！