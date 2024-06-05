import argparse
from solver import train_resnet18

def resnet_train():
    parser = argparse.ArgumentParser(description='Training Resnet-18 on the Tiny-ImageNet dataset')

    parser.add_argument('--epochs_resnet', type=int, default=30, help='Number of training epochs for ResNet-18')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training ResNet-18')
    parser.add_argument('--save', type=bool, default=False, help='Save the trained ResNet-18 model')

    parser.add_argument('--seed', type=int, default=605, help='Random seed for reproduceability')
    parser.add_argument('--root', type=str, default='./data/', help='Root directory of the data')
    parser.add_argument('--batch_size', type=int, default=64, help='Size of each batch during training')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers used in data loading')
    parser.add_argument('--lr_configs', type=dict, default={}, help='Additional configs for learning rate of ResNet-18')

    args = parser.parse_args()

    train_resnet18(
        epochs=args.epochs_resnet,
        lr=args.lr,
        save=args.save,
        seed=args.seed,
        data_root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr_configs=args.lr_configs,
    )

if __name__ == "__main__":
    resnet_train()