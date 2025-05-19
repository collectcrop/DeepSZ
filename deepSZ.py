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
    
    os.system("python ./extract_weights_and_compression.py")    #extract weights compress and decompress
    os.system("python ./reassemble_and_test.py 6")              #test on accuracy degradation with different compression ratio
    os.system("python ./reassemble_and_test.py 7")
    os.system("python ./reassemble_and_test.py 8")
    os.system("python ./optimize.py")                           #optimize compression config for each layer and reconstruct model