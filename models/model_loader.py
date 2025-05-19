import torch.nn as nn
from torchvision import models

def get_model(model_type, num_classes=10):
    if model_type == 'alexnet':
        model = models.alexnet()
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_type == 'vgg16':
        model = models.vgg16()
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_type == 'lenet5':
        from .lenet5 import LeNet5
        model = LeNet5()
    elif model_type == 'lenet-300-100':
        from .lenet300 import LeNet300100
        model = LeNet300100()
    else:
        raise ValueError(f"unsupported model: {model_type}")
    return model
