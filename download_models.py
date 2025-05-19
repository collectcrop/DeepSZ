import torch
import torchvision.models as models
import os

# 下载 AlexNet 预训练模型
print("Downloading AlexNet...")
alexnet = models.alexnet(pretrained=True)
torch.save(alexnet.state_dict(), "model/alexnet.pth")
print("AlexNet saved to model/alexnet.pth")

# 下载 VGG-16 预训练模型
print("Downloading VGG-16...")
vgg16 = models.vgg16(pretrained=True)
torch.save(vgg16.state_dict(), "model/vgg16.pth")
print("VGG-16 saved to model/vgg16.pth")

print("✅ All models downloaded and saved successfully.")
