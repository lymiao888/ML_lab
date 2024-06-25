# 迁移学习

## 作业内容
使用深度学习框架 Pytorch，在 ImageNet 预训练图像分类模型（例如 Resnet18）基础上，对自己的 30 类水果图像分类数据集进行迁移学习（transfer learning）微调（fine-tuning）训练，考虑三种迁移学习训练方式：
1. 只微调训练模型最后一层（全连接分类层）；
2. 微调训练所有层；
3. 随机初始化模型全部权重，从头训练所有层。得到自己的图像分类模型并保存。

## 作业说明
1. 目录结构
```
Task6
├── checkout
│   └── fruit30_pytorch_C1.pth
│   └── fruit30_pytorch_C2.pth
│   └── fruit30_pytorch_C3.pth
├── fruit30_split
│   └── train
│       └── ...
│   └── val
│       └── ...
├── README.md
└── src
    ├── fruit_all_layers.py
    ├── fruit_from_scratch.py
    ├── fruit_last_layer.py
    └── utils.py
```   
**核心文件:**
fruit_last_layer.py : 只微调训练模型最后一层（全连接分类层）；
fruit_all_layers.py : 微调训练所有层；
fruit_from_scratch.py : 随机初始化模型全部权重，从头训练所有层。得到自己的图像分类模型并保存

2. 下面以fruit_last_layer.py为例，运行代码的bash命令
```bash
cd Task6
wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/fruit30/fruit30_split.zip
mkdir checkpoint
cd ./src
python fruit_last_layer.py --data_dir "../fruit30_split" --batch_size 128 --epochs 60 --output_dir "../checkpoint/fruit30_pytorch_C1.pth"
```
3. 代码解释：
fruit_last_layer.py : 
```python
#!!! Freeze parameters in all layers and replace the last fully connected layer
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(train_loader.dataset.classes))
```
fruit_all_layers.py :
```python
#!!! Set requires_grad = True to fine-tune all layers.
    for param in model.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(train_loader.dataset.classes))
```
fruit_from_scratch.py : 
```python
#!!! Randomly initialize all weights by set pretrained=False
    model = models.resnet18(pretrained=False) 
    model.fc = nn.Linear(model.fc.in_features, len(train_loader.dataset.classes))
```

## 注：
1. resnet18-f37072fd.pth会被下载到.cache/torch, 可以 $ export TORCH_HOME="/path/to/download/location"

2. 关于matplotlib显示中文的问题，因为我的环境是python3.9所以下面的是3.9
```bash
wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf -O {$home/username}/miniconda3/envs/ML/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf --no-check-certificate
rm -rf {$home/username}/.cache/matplotlib
```