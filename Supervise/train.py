import yaml
import argparse
from solver import train_byol, train_resnet18


# get the default training configurations
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
BYOL_TRAIN_CONFIG = config['stream'] | config['common'] | config['train']['byol']
RESNET_TRAIN_CONFIG = config['stream'] | config['common'] | config['train']['resnet']

def byol(args):
    train_byol(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        update_rate=args.update_rate,
        save=args.save,
        seed=args.seed,
        data_root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay
    )

def resnet(args):
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

def train():
    parser = argparse.ArgumentParser(description='Training models on the Tiny-ImageNet dataset.')
    subparsers = parser.add_subparsers(dest='model', required=True, help='Choose a training method(model)')

    # BYOL training parser
    byol_parser = subparsers.add_parser('byol', help='Train with self-supervised model BYOL')
    
    byol_parser.add_argument('--epochs'     , type=int  , default=BYOL_TRAIN_CONFIG['epochs']       , help='Number of training epochs for BYOL')
    byol_parser.add_argument('--lr'         , type=float, default=BYOL_TRAIN_CONFIG['lr']           , help='Learning rate of the optimizer')
    byol_parser.add_argument('--hidden_dim' , type=int  , default=BYOL_TRAIN_CONFIG['hidden_dim']   , help='Dimension of the projection space')
    byol_parser.add_argument('--output_dim' , type=int  , default=BYOL_TRAIN_CONFIG['output_dim']   , help='Dimension of the prediction space')
    byol_parser.add_argument('--update_rate', type=float, default=BYOL_TRAIN_CONFIG['update_rate']  , help='Update rate of the target by moving average')
    
    byol_parser.add_argument('--save'       , action='store_true'                                   , help='Save the trained BYOL model')
    byol_parser.add_argument('--seed'       , type=int  , default=BYOL_TRAIN_CONFIG['seed']         , help='Random seed for reproducibility')
    byol_parser.add_argument('--root'       , type=str  , default=BYOL_TRAIN_CONFIG['root']         , help='Root directory of the data')
    byol_parser.add_argument('--batch_size' , type=int  , default=BYOL_TRAIN_CONFIG['batch_size']   , help='Size of each batch during training')
    byol_parser.add_argument('--num_workers', type=int  , default=BYOL_TRAIN_CONFIG['num_workers']  , help='Number of workers used in data loading')
    byol_parser.add_argument('--weight_decay',type=float, default=BYOL_TRAIN_CONFIG['weight_decay'] , help='Weight decay in the optimizer')
    
    byol_parser.set_defaults(func=byol)

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

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    train()