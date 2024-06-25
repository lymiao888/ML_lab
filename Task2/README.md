# 实验 全连接BP神经网络

## 作业内容
使用BP神经网络去完成回归任务，数据集可以使用波士顿房价预测数据集或者加州房价预测数据集，或者其他回归问题数据集。
模型可在上面BP神经网络框架上进行改进，或者使用其他网络模型实现，给出平均绝对误差（MAE），均方误差(MSE)性能指标。

## 作业说明
1. 目录结构
```
Task2
├── dataset
│   └── house.csv
├── README.md
├── scripts
│   ├── run_example.sh
│   └── run_homework.sh
└── src
    ├── example.py
    ├── homework.py
    ├── model.py
    └── utils.py
```
可以在任意目录位置执行
```bash
sh run_homework.sh
```
脚本中的参数可以更改调整
```bash
python -u "$script_dir/../src/homework.py" "$script_dir/../dataset/house.csv" \
        --batch_size 64 \
        --epochs 16 \
        > "$script_dir/../output/output_homework.txt" 
```

2. 数据集--加州房价
UCL：https://raw.githubusercontent.com/huangjia2019/house/master/house.csv
由于有时候会有网络连接问题，我直接下载在本地了

3. 代码说明
本题根据所给example进行改写，example是对 MNIST 手写数字数据集通过BP全联接神经网络进行分类任务；
根据作业要求，model.py文件中，我在fully_connected_model的基础上添加了RegressionModel作为回归模型；
同时加入参数的传入，可以灵活变动batch_size,epochs和data_dir；
在测试过程中，添加了平均绝对误差（MAE），均方误差(MSE)做性能指标方便观察。