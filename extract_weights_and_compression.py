import os
import sys
import numpy as np
import torch
from pathlib import Path
from models.model_loader import get_model

data_dir = Path("./data")
model_dir = Path("./model")
def load_model(model_name):
    # 加载模型
    if os.path.exists(model_dir / f'{model_name}.pth'):
        model = torch.load(model_dir / f'{model_name}.pth', map_location='cpu')  # 训练后保存的模型路径
        state_dict = model['state_dict'] if 'state_dict' in model else model
        print(f"model {model_name} loaded!")
        return state_dict
    else:
        print("model not exists!")
        os._exit(0)
        
def count_fc_params(model, model_name):
    total_params = 0

    if model_name in ["alexnet", "vgg16"]:
        for layer in model.classifier:
            if isinstance(layer, torch.nn.Linear):
                total_params += layer.weight.numel()
                if layer.bias is not None:
                    total_params += layer.bias.numel()

    elif model_name in ["lenet5", "lenet300"]:
        fc_layers = ['fc1', 'fc2', 'fc3']
        for name in fc_layers:
            layer = getattr(model, name, None)
            if layer is not None:
                total_params += layer.weight.numel()
                if layer.bias is not None:
                    total_params += layer.bias.numel()

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return total_params

def compress_fc_layer(tensor, layer_name, layer_idx):
    print(f"cope with layer {layer_name} ...")
    
    # 转换为 1D 向量
    tensor = tensor.cpu().numpy().reshape(-1)
    
    k = len(tensor)
    data = np.zeros(k, dtype='float32')
    index = np.zeros(k, dtype='uint8')

    k = 0           # index数组下标
    kk = 0          # data数组下标
    bit = 0         # 连续为0的个数

    for i in range(len(tensor)):
        if bit == 255 and tensor[i] == 0:
            index[k] = 0
            k += 1
            bit = 0
        if tensor[i] != 0:
            data[kk] = tensor[i]
            index[k] = bit
            k += 1
            kk += 1
            bit = 0
        bit += 1

    # 截取有效非零元素
    a = data[:kk]
    b = index[:k]

    # 稀疏矩阵表示
    a.astype('float32').tofile(f'./data/{layer_name}-data-o.dat')
    b.astype('uint8').tofile(f'./data/{layer_name}-index-o.dat')

    print(f"{layer_name} 提取完成，非零数：{kk}，索引数：{k}")

    print(f"{layer_name} 开始压缩 ...")
    os.system(f"python ./bash_script.py {kk} {layer_idx}")
    os.system(f"bash ./SZ_compress_script/{layer_name}_script.sh")
    print(f"{layer_name} 压缩完成！")

def compress_model_layers(state_dict, model_type):
    # 根据模型结构选择要压缩的层
    if model_type == 'alexnet':
        layer_info = {
            'fc6': ('classifier.1.weight', 6),
            'fc7': ('classifier.4.weight', 7),
            'fc8': ('classifier.6.weight', 8)
        }
    elif model_type == 'lenet-300-100':
        layer_info = {
            'fc1': ('fc1.weight', 1),
            'fc2': ('fc2.weight', 2),
            'fc3': ('fc3.weight', 3)
        }
    elif model_type == 'lenet5':
        layer_info = {
            'fc1': ('fc1.weight', 1),
            'fc2': ('fc2.weight', 2),
            'fc3': ('fc3.weight', 3)
        }
    elif model_type == 'vgg16':
        layer_info = {
            'fc6': ('classifier.0.weight', 6),
            'fc7': ('classifier.3.weight', 7),
            'fc8': ('classifier.6.weight', 8)
        }
    else:
        print(f"未知模型类型：{model_type}")
        sys.exit(1)

    for layer_name, (param_name, idx) in layer_info.items():
        tensor = state_dict.get(param_name, None)
        if tensor is not None:
            compress_fc_layer(tensor, layer_name, idx)
        else:
            print(f"waring: {param_name} not found in state_dict")
def main():
    if len(sys.argv) != 2:
        print("  please give valid model name as the parameter,for example:")
        print("  python compress.py alexnet")
        sys.exit(1)

    model_type = sys.argv[1].lower()

    if not os.path.exists('./data'):
        os.makedirs('./data')

    state_dict = load_model(model_type)
    if model_type == 'lenet5' or model_type == 'lenet300':
        num_classes = 10
    elif model_type == 'alexnet' or model_type == 'vgg16':
        num_classes = 200
    
    # 记录压缩前fc层总大小
    model = get_model(model_type,num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    total_params = count_fc_params(model, model_type)
    size = total_params * 32
    print(f"Total FC layer parameters (uncompressed): {total_params}")
    print(f"Size in bytes (float32): {total_params * 4} Bytes, or {total_params * 4 / 1024:.2f} KB")
    with open(f"{data_dir}/{model_type}-origin-fc-size.txt", "w") as f:
        f.write(f"{size}\n")
    # 开始压缩
    compress_model_layers(state_dict, model_type)

    print("all done!")

if __name__ == '__main__':
    main()