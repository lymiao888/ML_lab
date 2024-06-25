# 半监督学习

## 作业内容
基于 tutorial 中的内容，完成以下任务：
1. 对无标签集设计并应用一个熵正则化损失函数，调整该损失函数的权重，用于使模型输出尽可能确信的预测结果。
2. FixMatch 是一个半监督学习算法，对于无标签集中的样本，它将其中的弱增强图片的预测结果视为伪标签，然后筛选其中的高置信度样本，最后训练模型将对应的强增强图片的预测结果与伪标签对齐。请尝试根据这一思路实现 FixMatch 算法。


## 作业说明
1. 目录结构

2. 代码说明
   
L = labeled_loss + unlabeled_loss + lambda_entropy * entropy_regularization_loss(unlabeled_outputs)


