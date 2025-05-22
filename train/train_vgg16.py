import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# 1. 设置数据路径和超参数
data_dir = './data/tiny-imagenet-200'  
model_dir = './model'
num_classes = len(os.listdir(os.path.join(data_dir, 'train')))
batch_size = 32
num_epochs = 20
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 数据增强 & 预处理
normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize,
])

# 3. 加载数据集
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def main():
    # 4. 加载预训练 VGG-16 模型
    model = models.vgg16(pretrained=True)
    # 替换最后的分类器
    model.classifier[6] = nn.Linear(4096, num_classes)
    model = model.to(device)

    # 5. 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # 6. 训练函数
    def train(epoch):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"[Train Epoch {epoch}]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix(loss=running_loss/total, acc=100.*correct/total)

    # 7. 验证函数
    def validate():
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        return acc

    # 8. 主训练循环
    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        acc = validate()
        print(f"Validation Accuracy: {acc:.2f}%")

        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{model_dir}/vgg16.pth")
            print(">> Saved Model")

    print(f"Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    print("Training start!")
    import multiprocessing
    multiprocessing.freeze_support()
    main()
    print("Training complete ")