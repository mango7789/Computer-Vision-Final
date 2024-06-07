import os
import yaml
import argparse
from solver import test_with_model


# get the default training configurations
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
TEST_CONFIG = config['stream'] | config['test']

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test the trained CNN or ViT model on CIFAR-100 dataset.')
    
    parser.add_argument('--root', type=str, default=TEST_CONFIG['data_root'], help='The root directory of the dataset.' )
    parser.add_argument('--path', type=str, default=TEST_CONFIG['cnn_path'], help='The name of the model file, the file is assumed in the `./model` directory')
    
    args = parser.parse_args()
    
    test_with_model(
        data_root=args.root,
        path=os.path.join('./model', args.path)
    )