# Attrans: Attention Status Identification Using Transformer Models.
## Introduction
Attrans模型的输入是一段长度为250、通道数为4的神经信号，Attrans会将其识别为“Relaxed”，“Neutral”或“Concentrating”三种状态。该模型可下载到本地运行，建议配合我们的硬件设备使用。

Attrans模型的核心是一个预训练的Transformer模型。在本问题中，我们将Transformer应用于多通道时间（神经）信号是一个很自然的想法。Transformer架构以其在处理文本信息上的优越性而名噪一时，但它富有创造性的注意力机制的应用空间远不止上下文理解：将注意力机制用于处理时空信号的处理方法已经在Operator Learning领域内获得巨大成功。首先，注意力机制所赋予Transformer的积分特性使得它天然地比局域的、微分的传统机器学习更能提取到蕴藏在信号内部的长时段激发特征；其次，Transformer的模型机制使得它与以往的神经网络模型相比更少地收到梯度消失的影响，模型的有效参数增量得以大大提升，我们可以用大语言模型的优势跳过繁琐的传统分析（如FFT、Wavelet等）、直接使用大规模的嵌入空间表示和学习信号波的频率成分，并对此前的机器学习模型形成优势地位，这使得我们的模型更有机会窥见更为本质的信息、达到更好的效果。

为了在正式训练前使模型对神经信号这门“语言”具有一定的理解（可以理解为实质上是让模型预先学习到信号中具有的某些最为常见的、比较重要的特定的波成分并分配更多的“注意力”），我们使用了预训练技术。在预训练阶段，我们通过掩码操作，掩蔽掉信号的某些片段（不少于15%）并命令模型自行填充缺失的部分，计算损失并学习修正，使得模型学习到了一些关于该种信号的“背景知识”。实验表明，在使用预训练技术后，分类效果显著好于不使用预训练的模型。

我们设计了灵活的模型接口，使得用户可以自由地在GPT-2、BERT、RoBERTa等模型的各种版本之间进行切换，仅需更改config中的model_name。经过消融实验，我们得出结论，GPT-2类型的Transformer模型最适合解决该类问题。原因如下：
* GPT模型作为当前较新的语言模型，它的注意力全部来自于左侧文段。对于我们的分类场景，Transformer模型的核心任务实际上是提取信号成分、降维参数空间，以便更好地分门别类。因此我们不需要双向注意力机制和掩码操作，GPT-2的注意力足以将全局信息聚焦在它最后一个输出的token上，精简了注意力结构，提升了模型效率。
* GPT模型具有显著的平行作用效果和积分特性，适合在具有某种”平移不变性“的数据上进行平行训练。当前的该方面数据集多为长度远长于250的信号，且该长信号往往代表同一注意力类型，因此GPT-2的平行作用效果可以很好地适配该数据类型进行针对性训练。

为了更充分地利用数据，我们在构造训练集和测试集时使用了有重叠的采样。

## Quick Start
运行```python main.py pretrain```即可开始预训练。可以在main.py文件中修改训练集路径和初始化模型权重。

运行```python main finetune```即可开始微调（训练分类器）。可以在main.py中使用预训练的模型初始化分类器的权重。

## Result
我们使用数据的20%作为测试集，80%作为训练集，在训练过程中只使用训练集进行训练，并在测试集上测试结果。我们的模型在预训练和微调中收敛效果良好，模型准确度达到了95.99%，具有较高的稳定性。
