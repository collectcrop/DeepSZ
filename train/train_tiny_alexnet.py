import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# ========= 配置参数 ========= #
data_dir = "./data/tiny-imagenet-200"
num_classes = 200
batch_size = 64
num_epochs = 50
learning_rate = 1e-4
save_path = "./model/alexnet_tiny.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 数据增强 & 加载 ========= #
normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
transform_train = transforms.Compose(
    [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        normalize, ])
transform_val = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize, ])

train_dataset = ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
val_dataset = ImageFolder(os.path.join(data_dir, "val"), transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def main():
    # ========= 模型准备 ========= #
    model = torchvision.models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(4096, num_classes)
    model = model.to(device)

    # ========= 损失函数与优化器 ========= #
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ========= 训练 & 验证循环 ========= #
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train
        print(f"Epoch {epoch+1} - Train Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 验证
        model.eval()
        correct_val = total_val = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        print(f"Epoch {epoch+1} - Val Acc: {val_acc:.4f}")

        # 每轮保存模型
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    
if __name__ == "__main__":
    print("Training start!")
    import multiprocessing
    multiprocessing.freeze_support()
    main()
    print("Training complete ")
