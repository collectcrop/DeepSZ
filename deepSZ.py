import os
import sys
import argparse
from pathlib import Path

models  = [
    'lenet5',
    'lenet-300-100',
    'alexnet',
    'vgg16'
]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=models, help="Model name")
    parser.add_argument('--data_dir', type=str, default="./data", help="Dataset directory")
    parser.add_argument('--model_dir', type=str, default="./model", help="Directory to load base model")
    parser.add_argument('--output_dir', type=str, default="./decompressed_model", help="Output directory for modified models")
    args = parser.parse_args()
    
    model_name = args.model
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    
    os.system(f"python ./extract_weights_and_compression.py {model_name}")    #extract weights compress and decompress
    if model_name == 'lenet5' or model_name == 'lenet-300-100':
        r = range(1,4)
    elif model_name == 'alexnet' or model_name == 'vgg16':
        r = range(6,9)
    for i in r:
        bashline = f"python ./reassemble_and_test.py --model {model_name} --layer {i} --data_dir {data_dir} --model_dir {model_dir} --output_dir {output_dir}"
        os.system(bashline)
    os.system(f"python ./optimize.py {model_name}")                           #optimize compression config for each layer and reconstruct model
    
if __name__ == "__main__":
    main()