import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
from tqdm import tqdm
from timeit import default_timer as timer 
from utils import build_train_dataloader, build_test_dataloader, print_train_time, accuracy_fn


def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', type=str, default='../fruit30_split', help='path to the data file')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for processing (default: 64)')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs for processing (default: 80)')
    parser.add_argument('--output_dir', type=str, default='../checkpoint/fruit30_pytorch_C2.pth', help='path to the output checkpoint')
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    dataset_dir = args.data_dir
    output_dir = args.output_dir

    # build dataloader
    train_loader = build_train_dataloader(dataset_dir, batch_size)
    test_loader = build_test_dataloader(dataset_dir, batch_size)

    # load the pretrain 
    model = models.resnet18(pretrained=True)   
    
    #!!! Set requires_grad = True to fine-tune all layers.
    for param in model.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(train_loader.dataset.classes))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    train_time_start_on_cpu = timer()
    best_test_acc = 0.0

    for epoch in tqdm(range(epochs)):
        # ----- train -----    
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)

        # ----- test -----      
        test_loss, test_acc = 0.0, 0.0 
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                test_pred = model(X)
                test_loss += criterion(test_pred, y).item()
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(test_loader.dataset)
        test_acc = (test_acc / len(test_loader))

        print(f"Epoch {epoch+1}| Train Loss: {train_loss:.4f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")
        scheduler.step()
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), output_dir)
    train_time_end_on_cpu = timer()
    print_train_time(
        start=train_time_start_on_cpu, 
        end=train_time_end_on_cpu,
        device=str(next(model.parameters()).device)
    )
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")

if __name__ == "__main__":
    main()