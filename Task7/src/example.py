import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, Dataset
import torch.nn.functional as F
import numpy as np
from dataset import UnlabeledDataset
from utils import *

def main():
    
    labeled_size = 10000 
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # build dataset
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    full_dataset = datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./', train=False, download=True, transform=transform)

    # Training Set: Randomly select some as the labeled set, others as the unlabeled set.
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    labeled_indices = indices[:labeled_size]
    unlabeled_indices = indices[labeled_size:]
    labeled_dataset = Subset(full_dataset, labeled_indices)
    unlabeled_dataset = UnlabeledDataset(Subset(full_dataset, unlabeled_indices))

    # build data_loader
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model init
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10) # CIFAR10 have 10 classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    labeled_loss_history = []
    unlabeled_loss_history = []
    test_accuracy_history = []

    for epoch in range(num_epochs):
        unlabeled_train_iter = iter(unlabeled_loader)
        # Outer loop : labeled dataset
        for batch_idx, batch_x in enumerate(labeled_loader): 
            # Inner loop : unlabeled dataset
            try:   
                batch_u = next(unlabeled_train_iter)
            # If the unlabeled dataset iteration is complete, restart a new loop!
            except StopIteration:  
                unlabeled_train_iter = iter(unlabeled_loader)
                batch_u = next(unlabeled_train_iter)

            labeled_loss = labeled_train(batch_x, model, criterion, device)
            unlabeled_loss = unlabeled_train(batch_u, model, device)

            L = labeled_loss + unlabeled_loss
            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            labeled_loss = labeled_loss.item()
            unlabeled_loss = unlabeled_loss.item()
            labeled_loss_history.append(labeled_loss)
            unlabeled_loss_history.append(unlabeled_loss)

            if batch_idx % 100 == 0:
                print("Epoch: {}, Iter: {}, Labeled Loss: {}".format(epoch, batch_idx, labeled_loss))
                print("Epoch: {}, Iter: {}, Unlabeled Loss: {}".format(epoch, batch_idx, unlabeled_loss))

        test_accuracy = test(test_loader, model, device)
        test_accuracy_history.append(test_accuracy)
        print("Epoch: {}, Iter: {}, Test Accuracy: {}".format(epoch, batch_idx, test_accuracy))

if __name__ == "__main__":
    main()
