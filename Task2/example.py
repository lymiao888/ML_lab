import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
from timeit import default_timer as timer 
from model import fully_connected_model
from utils import print_train_time, accuracy_fn

def main():
    # Print version information
    print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

    BATCH_SIZE = 32
    EPOCHS = 5

    # Load MNIST dataset
    train_data = torchvision.datasets.MNIST(
        root='MNIST',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_data = torchvision.datasets.MNIST(
        root='MNIST',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    train_losses = []
    test_losses = []
    test_accuracies = []

    torch.manual_seed(42)

    # Initialize model
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

        # ----- train ----- 
        train_loss = 0
        for batch, (X, y) in enumerate(train_dataloader):
            model.train() 
            y_pred = model(X) # Forward pass
            loss = loss_function(y_pred, y) # Compute loss (per batch)
            train_loss += loss.item() # Accumulate loss over each epoch
            optimizer.zero_grad() # Zero the gradients of the optimizer
            loss.backward() # Backward pass for loss
            optimizer.step() # Optimizer step to update parameters
        train_loss /= len(train_dataloader)

        # ----- test ----- 
        test_loss, test_acc = 0, 0 
        model.eval()
        with torch.no_grad():
            for X, y in test_dataloader:
                test_pred = model(X) # Forward pass
                test_loss += loss_function(test_pred, y).item() # Compute loss and Accumulate loss over each epoch
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1)) # Compute accuracy
            test_loss /= len(test_dataloader)
            test_acc = (test_acc / len(test_dataloader)) * 100  # Calculate accuracy percentage

        ## Print training status
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    # Calculate and print total training time
    train_time_end_on_cpu = timer()
    total_train_time_model = print_train_time(
        start=train_time_start_on_cpu, 
        end=train_time_end_on_cpu,
        device=str(next(model.parameters()).device)
    )

if __name__ == "__main__":
    main()