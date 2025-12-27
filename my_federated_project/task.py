"""
task.py: Shared resources for the Federated Learning project.

This module acts as the 'contract' between ClientApp and ServerApp. It contains:
1. The Neural Network definition (Net).
2. Data loading and partitioning logic.
3. The shared 'TrainProcessMetadata' dataclass used for custom communication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr.app import ArrayRecord, MetricRecord
from dataclasses import dataclass

# =============================================================================
# 1. Device Selection & Model Definition
# =============================================================================

def get_device() -> torch.device:
    """Selects the best available device (CUDA, MPS, or CPU) for training."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

class Net(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for CIFAR-10 classification.
    Adapted from 'PyTorch: A 60 Minute Blitz'.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# =============================================================================
# 2. Data Loading & Partitioning
# =============================================================================

fds = None  # Global cache for FederatedDataset to prevent reloading per client
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def apply_transforms(batch):
    """Apply normalization and tensor conversion to a batch of images."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_data(partition_id: int, num_partitions: int):
    """
    Load a specific partition of the CIFAR-10 dataset.
    
    Args:
        partition_id (int): The unique ID of the current client node.
        num_partitions (int): Total number of partitions (clients) in the federation.
        
    Returns:
        tuple: (trainloader, testloader) for the specific partition.
    """
    global fds
    if fds is None:
        # Use IidPartitioner to distribute data evenly and randomly across clients
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    
    # Load the specific chunk of data for this client
    partition = fds.load_partition(partition_id)
    
    # Split local data: 80% for training, 20% for local validation
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    
    return trainloader, testloader

def load_centralized_dataset():
    """Load the full CIFAR-10 test set for centralized evaluation on the Server."""
    centralized_fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={})
    full_test_dataset = centralized_fds.load_split("test").with_transform(apply_transforms)
    return DataLoader(full_test_dataset, batch_size=32, shuffle=False)

# =============================================================================
# 3. Training & Evaluation Logic
# =============================================================================

def train(net, trainloader, epochs, lr, device) -> tuple[float, dict[str, float]]:
    """
    Train the model on the training set.

    Returns:
        avg_loss (float): The average loss of the final epoch.
        epoch_losses (dict): A history of average loss per epoch (e.g., {'epoch_1': 0.5, ...}).
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    
    epoch_losses: dict[str, float] = {} 
    avg_loss = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()         # Reset gradients
            outputs = net(images)         # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()               # Backpropagation
            optimizer.step()              # Update weights
            
            running_loss += loss.item()
        
        # Calculate average loss for this specific epoch
        avg_loss = running_loss / len(trainloader)
        epoch_losses[f"epoch_{epoch+1}"] = avg_loss
        
    return avg_loss, epoch_losses 

def test(net, testloader, device) -> tuple[float, float]:
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy    

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate the global model on the centralized test set (Server-side)."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    model.to(device)
    test_loader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_loader, device)
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})

# =============================================================================
# 4. Custom Metadata Contract
# =============================================================================

@dataclass
class TrainProcessMetadata:
    """
    DataClass to hold custom metadata about the training process.
    This object is serialized by the Client and sent to the Server.
    """
    training_time: float
    converged: bool
    training_losses: dict[str, float]