import os
import sys
import torch
import torch.nn as nn
import numpy as np
import subprocess
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.model_loader import get_model
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
# ==== 精度评估函数 ====
def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

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
# ==== 主函数 ====
def main():
    # 参数设置
    model_name = sys.argv[1]  # 'alexnet', 'vgg16', 'lenet5', 'lenet-300-100'
    data_dir = "./data"
    model_dir = "./model"
    output_dir = "./decompressed_model"
    os.makedirs(output_dir, exist_ok=True)

    # 层结构定义
    if model_name in ['alexnet', 'vgg16']:
        layer_shapes = {
            "6": (4096, 9216),
            "7": (4096, 4096),
            "8": (200, 4096),  # 已从1000类别改成200类别
        }
        layer_indices = {
            "6": 1,
            "7": 4,
            "8": 6,
        }
        num_classes = 200
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])
        val_dataset = datasets.ImageFolder(f"{data_dir}/tiny-imagenet-200/val", transform=transform)
    
    elif model_name in ['lenet5', 'lenet-300-100']:
        layer_shapes = {
            "1": (120, 256),  
            "2": (84, 120),
            "3": (10, 84),
        }
        layer_indices = {
            "1": 4,
            "2": 6,
            "3": 8,
        }
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        val_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{model_name}.pth")))
    model.eval()

    o_accuracy = evaluate(model, val_loader, device)
    print(f"original accuracy: {o_accuracy:.4f}")
    origin_size = 0
    with open(f"{data_dir}/{model_name}-origin-fc-size.txt", "r") as f:
        origin_size = int(f.readline())
    print(f"original size: {origin_size}")

    e_accuracy = 0.005  # 可接受的精度损失
    max_test = 9    
    num_layers = len(layer_shapes)          # 要压缩的层数
    delta = np.zeros((num_layers, max_test), dtype='float32')
    ratio = np.zeros((num_layers, max_test), dtype='float32')

    # 计算精度损失和压缩率
    for idx, layer_num in enumerate(layer_shapes.keys()):
        accuracy = np.fromfile(f"{data_dir}/fc{layer_num}-accuracy.txt", dtype='float32')
        for i in range(max_test):
            delta[idx][i] = o_accuracy - accuracy[i+1]
        read_line = open(f"{data_dir}/compression_ratios_fc{layer_num}.txt").read().split()
        print(read_line)
        for i in range(max_test):
            ratio[idx][i] = float(read_line[i*3+2])

    min_size = sum(ratio[i][0] for i in range(num_layers))
    best_indices = [0] * num_layers

    # 搜索最佳压缩配置
    for indices in np.ndindex(*(max_test,) * num_layers):
        total_delta = sum(delta[i][indices[i]] for i in range(num_layers))
        total_ratio = sum(ratio[i][indices[i]] for i in range(num_layers))
        if total_delta <= e_accuracy and total_ratio < min_size:
            min_size = total_ratio
            best_indices = list(indices)

    # 替换模型层的参数
    for idx, layer_num in enumerate(layer_shapes.keys()):
        i = best_indices[idx]
        data_file = os.path.join(data_dir, f"fc{layer_num}-data-{i+1}E-3.dat")
        index_file = os.path.join(data_dir, f"fc{layer_num}-index-o.dat")
        feat = decompress_weights(data_file, index_file, layer_shapes[layer_num])
        weight_tensor = torch.tensor(feat, dtype=torch.float32)
        layer_idx = layer_indices[layer_num]
        replace_weights(model, model_name, layer_idx, weight_tensor, device)
        

    # 保存新模型
    model_path = os.path.join(output_dir, f"{model_name}_compressed.pth")
    torch.save(model.state_dict(), model_path)

    # 评估新模型精度
    acc = evaluate(model, val_loader, device)
    print(f"accuracy after compression: {acc:.4f}")
    print(f"best error bounds configure: {best_indices}")
    print(f"size after compression: {min_size:.4f}")
    print(f"compression ratio: {min_size*100/origin_size:.4f}%")

if __name__ == "__main__":
    main()
