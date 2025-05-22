import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import datasets, transforms, models
import torchvision
from torch.utils.data import DataLoader
from models.model_loader import get_model
import argparse

data_dir = Path("./data")
model_dir = Path("./model")
output_dir = Path("./decompressed_model")
output_dir.mkdir(exist_ok=True)

MODEL_CONFIGS = {
    "alexnet": {
        "layer_shapes": {
            "6": (4096, 9216),
            "7": (4096, 4096),
            "8": (200, 4096),  # 200类输出
        },
        "layer_indices": {
            "6": 1,
            "7": 4,
            "8": 6,
        }
    },
    "vgg16": {
        "layer_shapes": {
            "6": (4096, 25088),
            "7": (4096, 4096),
            "8": (200, 4096),
        },
        "layer_indices": {
            "6": 0,
            "7": 3,
            "8": 6,
        }
    },
    "lenet5":{
        "layer_shapes": {
            "1": (120, 256),
            "2": (84, 120),
            "3": (10, 84),
        },
        "layer_indices": {
            "1": 4,
            "2": 6,
            "3": 8,
        }
    }
}
# ==== 解压稀疏权重函数 ====
def decompress_weights(data_path, index_path, shape):
    data = np.fromfile(data_path, dtype='float32')
    index = np.fromfile(index_path, dtype='uint8')
    feat = np.zeros(shape[0] * shape[1], dtype='float32')

    k = q = 0
    for j in range(len(index)):
        if index[j] == 0 and j != 0:
            k += 255
        else:
            k += int(index[j])
            feat[k] = data[q]
            q += 1

    return np.reshape(feat, shape)

def replace_weights(model, model_name, layer_idx, weight_tensor, device):
    # 替换对应层权重
    if model_name == "alexnet" or model_name == "vgg16":
        model.classifier[layer_idx].weight.data = weight_tensor.to(device)
    elif model_name == "lenet5" or model_name == "lenet300":
        # 映射 layer_idx 到属性名
        layer_map = {
            4: 'fc1',
            6: 'fc2',
            8: 'fc3',
        }
        layer_name = layer_map.get(layer_idx)
        if layer_name is None:
            raise ValueError(f"Invalid layer index {layer_idx} for model {model_name}")

        layer_module = getattr(model, layer_name)
        if weight_tensor.shape != layer_module.weight.data.shape:
            raise ValueError(f"Weight shape mismatch: expected {layer_module.weight.data.shape}, got {weight_tensor.shape}")

        layer_module.weight.data = weight_tensor.to(device)
# ==== 主函数入口 ====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=MODEL_CONFIGS.keys(), help="Model name")
    parser.add_argument('--layer', type=str, required=True, help="FC layer number")
    parser.add_argument('--data_dir', type=str, default="./data", help="Dataset directory")
    parser.add_argument('--model_dir', type=str, default="./model", help="Directory to load base model")
    parser.add_argument('--output_dir', type=str, default="./decompressed_model", help="Output directory for modified models")
    args = parser.parse_args()
    
    model_name = args.model
    layer_num = args.layer
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    config = MODEL_CONFIGS[model_name]
    if layer_num not in config["layer_shapes"]:
        raise ValueError(f"invalid layer: {layer_num}")
    x, y = config["layer_shapes"][layer_num]
    layer_idx = config["layer_indices"][layer_num]
    
    # 加载模型
    if model_name == 'lenet5' or model_name == 'lenet300':
        num_classes = 10
    elif model_name == 'alexnet' or model_name == 'vgg16':
        num_classes = 200
        
    print(f"Loding {model_name} ...")
    model = get_model(model_name,num_classes)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pth"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 准备验证集
    print("Loding test dataset...")
    if model_name == "alexnet" or model_name == "vgg16":
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])
        val_dir = data_dir / "tiny-imagenet-200/val"
        if not val_dir.exists():
            raise FileNotFoundError(f"test dataset not exist: {val_dir}")
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    elif model_name == "lenet5" or model_name == "lenet300":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=False)
    
    # 精度记录
    accuracy = np.zeros(11, dtype='float32')

    for i in range(1, 10):
        print(f"\nProcessing fc{layer_num} - {i}E-3")

        data_file = data_dir / f"fc{layer_num}-data-{i}E-3.dat"
        index_file = data_dir / f"fc{layer_num}-index-o.dat"

        if not data_file.exists() or not index_file.exists():
            print(f"压缩文件不存在: {data_file} 或 {index_file}")
            continue

        feat = decompress_weights(data_file, index_file, (x, y))
        weight_tensor = torch.tensor(feat, dtype=torch.float32)

        replace_weights(model, model_name, layer_idx, weight_tensor, device)

        # 保存模型
        model_path = output_dir / f"{model_name}_fc{layer_num}_{i}E-3.pth"
        torch.save(model.state_dict(), model_path)

        # 评估准确率
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Testing fc{layer_num} - {i}E-3"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        accuracy[i] = acc
        print(f"Accuracy for fc{layer_num} - {i}E-3: {acc:.4f}")

    # 保存准确率结果
    acc_file = data_dir / f"fc{layer_num}-accuracy.txt"
    accuracy.astype('float32').tofile(acc_file)
    print(f"\nAccuracy results saved to {acc_file}")

if __name__ == "__main__":
    main()
