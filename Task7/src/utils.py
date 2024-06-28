import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from PIL import ImageFilter
import torchvision.transforms as transforms

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
        
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def entropy_regularization_loss(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    return torch.mean(entropy)

def entropy_loss(logits):
    p_softmax = nn.functional.softmax(logits, dim=1)
    log_p_softmax = nn.functional.log_softmax(logits, dim=1)
    p_logp = p_softmax * log_p_softmax
    return -p_logp.sum(dim=1).mean()

def labeled_train(batch_x, model, criterion, device):
    model.train()
    images, labels = batch_x[0].to(device), batch_x[1].to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    return loss

def unlabeled_train(batch_u, model, device, temperature = 0.9, threshold = 0.9):
    model.train()
    images = batch_u.to(device)
    loss = torch.tensor(0.0, device = device)

    with torch.no_grad(): 
        outputs = model(images)
        probas = torch.softmax(outputs / temperature, dim = -1)
        max_probas, pseudo_labels = torch.max(probas, dim = -1)  
        high_confidence_mask = max_probas.ge(threshold).float()  

    if high_confidence_mask.sum() > 1:
        outputs = model(images)
        # only high_confidence_mask
        pseudo_loss = F.cross_entropy(outputs, pseudo_labels, reduction='none') * high_confidence_mask  
        loss = pseudo_loss.mean()
    return loss

def FixMatch_unlabeled_train(batch_u_weak, batch_u_strong, model, device, temperature=0.9, threshold=0.95):
    model.train()
    images_weak = batch_u_weak.to(device)
    images_strong = batch_u_strong.to(device)
    loss = torch.tensor(0.0, device = device)
    # Forward pass for weakly augmented images
    with torch.no_grad():
        outputs_weak = model(images_weak)
        probas_weak = torch.softmax(outputs_weak / temperature, dim=-1)
        max_probas_weak, pseudo_labels = torch.max(probas_weak, dim=-1)
        high_confidence_mask = max_probas_weak.ge(threshold).float()
    
    if high_confidence_mask.sum() > 0:
        # Forward pass for strongly augmented images
        outputs_strong = model(images_strong)
        
        # Calculate pseudo loss only for high confidence samples
        pseudo_loss = F.cross_entropy(outputs_strong, pseudo_labels, reduction='none')
        pseudo_loss = (pseudo_loss * high_confidence_mask).mean()
    else:
        pseudo_loss = torch.tensor(0.0, device=device)
        loss = pseudo_loss.mean()
    return loss

def test(test_loader, model, device):
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
