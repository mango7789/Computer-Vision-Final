import yaml
import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from utils import get_cifar_100_dataloader
from solver import train_simclr, fine_tune_resnet18, train_resnet18, extract_features, train_linear_classifier

import warnings
warnings.filterwarnings('ignore')

# get the default training configurations
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
SIMCLR_TRAIN_CONFIG = config['stream'] | config['common'] | config['train']['simclr']
RESNET_TRAIN_CONFIG = config['stream'] | config['common'] | config['train']['resnet']
LINEAR_TRAIN_CONFIG = config['common'] | config['train']['linear']

# subfunctions

def simclr(args):
    train_simclr(
        epochs=args.epochs,
        lr=args.lr,
        save=args.save,
        temperature=args.temp,
        seed=args.seed,
        data_root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay
    )

def resnet(args):
    fine_tune_resnet18(
        epochs=args.epochs,
        lr=args.lr,
        save=args.save,
        seed=args.seed,
        data_root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay
    )
    train_resnet18(
        epochs=args.epochs,
        lr=args.lr,
        save=args.save,
        seed=args.seed,
        data_root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay
    )
    
def linear(args):
    model = resnet50()
    model.fc = nn.Identity()
    model.load_state_dict(torch.load(args.model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_loader, test_loader = get_cifar_100_dataloader()
    train_feature, train_label = extract_features(model, train_loader)
    test_feature, test_label = extract_features(model, test_loader)
    train_linear_classifier(
        train_feature, train_label, test_feature, test_label, 
        epochs=args.epochs, learning_rate=args.lr,
        type=args.type, save=args.save
    )


# main parser

def train():
    parser = argparse.ArgumentParser(description='Training models on the Tiny-ImageNet dataset.')
    subparsers = parser.add_subparsers(dest='model', required=True, help='Choose a training method (model)')

    # simclr training parser
    simclr_parser = subparsers.add_parser('simclr', help='Train with self-supervised model simclr')
    
    simclr_parser.add_argument('--epochs'     , type=int  , default=SIMCLR_TRAIN_CONFIG['epochs']       , help='Number of training epochs for simclr')
    simclr_parser.add_argument('--lr'         , type=float, default=SIMCLR_TRAIN_CONFIG['lr']           , help='Learning rate of the optimizer')
    simclr_parser.add_argument('--temp'       , type=float, default=SIMCLR_TRAIN_CONFIG['temp']         , help='Temperature of the NCE loss')
    
    simclr_parser.add_argument('--save'       , action='store_true'                                   , help='Save the trained simclr model')
    simclr_parser.add_argument('--seed'       , type=int  , default=SIMCLR_TRAIN_CONFIG['seed']         , help='Random seed for reproducibility')
    simclr_parser.add_argument('--root'       , type=str  , default=SIMCLR_TRAIN_CONFIG['root']         , help='Root directory of the data')
    simclr_parser.add_argument('--batch_size' , type=int  , default=SIMCLR_TRAIN_CONFIG['batch_size']   , help='Size of each batch during training')
    simclr_parser.add_argument('--num_workers', type=int  , default=SIMCLR_TRAIN_CONFIG['num_workers']  , help='Number of workers used in data loading')
    simclr_parser.add_argument('--weight_decay',type=float, default=SIMCLR_TRAIN_CONFIG['weight_decay'] , help='Weight decay in the optimizer')
    
    simclr_parser.set_defaults(func=simclr)

    # ResNet-18 training parser
    resnet_parser = subparsers.add_parser('resnet', help='Train with supervised model ResNet-18')
    
    resnet_parser.add_argument('--epochs'       , type=int  , default=RESNET_TRAIN_CONFIG['epochs']     , help='Number of training epochs for ResNet-18')
    resnet_parser.add_argument('--lr'           , type=float, default=RESNET_TRAIN_CONFIG['lr']         , help='Learning rate of the optimizer')
    resnet_parser.add_argument('--save'         , action='store_true'                                   , help='Save the trained ResNet-18 model')
    resnet_parser.add_argument('--seed'         , type=int  , default=RESNET_TRAIN_CONFIG['seed']       , help='Random seed for reproducibility')
    
    resnet_parser.add_argument('--root'         , type=str  , default=RESNET_TRAIN_CONFIG['root']       , help='Root directory of the data')
    resnet_parser.add_argument('--batch_size'   , type=int  , default=RESNET_TRAIN_CONFIG['batch_size'] , help='Size of each batch during training')
    resnet_parser.add_argument('--num_workers'  , type=int  , default=RESNET_TRAIN_CONFIG['num_workers'], help='Number of workers used in data loading')
    resnet_parser.add_argument('--weight_decay' , type=float, default=RESNET_TRAIN_CONFIG['weight_decay'], help='Weight decay in the optimizer')

    
    resnet_parser.set_defaults(func=resnet)
    
    # Linear Classifier training parser
    linear_parser = subparsers.add_parser('linear', help='Train the linear classifier for the features')
    
    linear_parser.add_argument('--epochs'       , type=int  , default=LINEAR_TRAIN_CONFIG['epochs']     , help='Number of training epochs for ResNet-18')
    linear_parser.add_argument('--lr'           , type=float, default=LINEAR_TRAIN_CONFIG['lr']         , help='Learning rate of the optimizer')
    
    linear_parser.add_argument('--model'        , type=str  , default='./model/simclr.pth', help='Path to the trained model')
    linear_parser.add_argument('--type'         , type=str  , choices=['self_supervise', 'supervise_with_pretrain', 'supervise_no_pretrain'], help='Types of training')
    linear_parser.add_argument('--save'         , action='store_true'                                   , help='Save the trained linear model')

    linear_parser.set_defaults(func=linear)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    train()
