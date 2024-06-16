import argparse
from solver import test_trained_model

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test the trained CNN or ViT model on CIFAR-100 dataset.')
    
    parser.add_argument('--encoder', type=str, help='The path of the parameters of resnet18 encoder.' )
    parser.add_argument('--classifier', type=str, help='The path of the parameters of the linear classifier.')
    
    args = parser.parse_args()
    
    test_trained_model(
        encoder_path=args.encoder,
        classifier_path=args.classifier
    )