import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from models.model_loader import get_model

def load_model(model_name):
    path = f'model/{model_name}.pth'
    if not torch.cuda.is_available():
        map_location = 'cpu'
    else:
        map_location = 'cuda'

    if not (os.path.exists(path)):
        print(f"model {model_name} not exists!")
        sys.exit(1)

    model = torch.load(path, map_location=map_location)
    if 'state_dict' in model:
        state_dict = model['state_dict']
    else:
        state_dict = model
    return state_dict

def get_test_loader(model_type):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if model_type in ['alexnet', 'vgg16']:
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize, ])
        dataset = datasets.ImageFolder('./data/tiny-imagenet-200/val', transform=transform)
    else:
        dataset = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transform)

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    return loader

def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)
    correct = total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    print(f"test accuracy : {acc:.2f}%")
    return acc

def main():
    if len(sys.argv) != 2:
        print("usage: python evaluate.py <module name>")
        sys.exit(1)

    model_type = sys.argv[1].lower()
    num_classes = 200 if model_type in ['alexnet', 'vgg16'] else 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"load model {model_type} ...")
    state_dict = load_model(model_type)
    model = get_model(model_type, num_classes)
    model.load_state_dict(state_dict)

    print("load test data ...")
    test_loader = get_test_loader(model_type)

    print("start evaluating ...")
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()
