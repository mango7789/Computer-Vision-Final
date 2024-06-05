import argparse
from solver import train_byol

def BYOL_train():
    parser = argparse.ArgumentParser(description='Training BYOL on the Tiny-ImageNet dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of the optimizer')
    parser.add_argument('--hidden_dim', type=int, default=4096, help='Dimension of the projection space')
    parser.add_argument('--output_dim', type=int, default=256, help='Dimension of the prediction space')
    parser.add_argument('--update_rate', type=float, default=0.996, help='Update rate of the target by moving average')
    parser.add_argument('--save', type=bool, default=False, help='Save model flag')
    parser.add_argument('--seed', type=int, default=603, help='Random seed for reproduceability')
    parser.add_argument('--data_root', type=str, default='./data/', help='Root directory of the data')
    parser.add_argument('--batch_size', type=int, default=64, help='Size of each batch during training')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers used in data loading')
    parser.add_argument('--lr_configs', type=dict, default={}, help='Additional configs for learning rate')

    args = parser.parse_args()

    train_byol(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        update_rate=args.update_rate,
        save=args.save,
        seed=args.seed,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr_configs=args.lr_configs,
    )

if __name__ == "__main__":
    BYOL_train()