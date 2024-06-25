import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision import datasets

from tqdm import tqdm
import matplotlib.pyplot as plt
from timeit import default_timer as timer 

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

class fully_connected_model(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # Flatten the input
            nn.Linear(in_features=input_shape, out_features=hidden_units),  # in_features = number of features in the data sample (784 pixels)
            nn.Linear(in_features=hidden_units, out_features=output_shape)  # out_features = number of classes for classification (10)
        )
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

# print version
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

BATCH_SIZE = 32
EPOCHS = 5

train_data=torchvision.datasets.MNIST(
    root='MNIST',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
test_data=torchvision.datasets.MNIST(
    root='MNIST',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch
    shuffle=True # shuffle data every epoch
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)

train_losses = []
test_losses = []
test_accuracies = []

torch.manual_seed(42)

# init model
model = fully_connected_model(
    input_shape=784,  # input: 28x28=784
    hidden_units=10,
    output_shape=10
)
model.to("cpu")

loss_function = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)  

train_time_start_on_cpu = timer()

for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}\n-------")
    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        model.train() 
        # 1. 前向传播
        y_pred = model(X)

        # 2. 计算损失（每个批次）
        loss = loss_function(y_pred, y)
        train_loss += loss # 累计每个epoch的损失

        # 3. 优化器梯度清零
        optimizer.zero_grad()

        # 4. 反向传播损失
        loss.backward()

        # 5. 优化器更新参数
        optimizer.step()
    # 训练损失除以训练数据加载器的长度（每个批次每个epoch的平均损失）
    train_loss /= len(train_dataloader)

    # 设置变量用于累计损失和准确率
    test_loss, test_acc = 0, 0 
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. 前向传播
            test_pred = model(X)
           
            # 2. 计算损失（累计）
            test_loss += loss_function(test_pred, y) # 累计每个epoch的损失

            # 3. 计算准确率
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
        # 测试损失除以测试数据加载器的长度（每个批次）
        test_loss /= len(test_dataloader)

        # 测试准确率除以测试数据加载器的长度（每个批次）
        test_acc /= len(test_dataloader)

    ## 打印训练情况
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")
    train_losses.append(train_loss.detach().numpy())
    test_losses.append(test_loss.detach().numpy())
    test_accuracies.append(test_acc)
     
train_time_end_on_cpu = timer()
total_train_time_model = print_train_time(
    start=train_time_start_on_cpu, 
    end=train_time_end_on_cpu,
    device=str(next(model.parameters()).device)
)