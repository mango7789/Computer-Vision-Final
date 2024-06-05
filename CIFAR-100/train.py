import yaml
import argparse
from solver import train_with_params

# get the default training configurations
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
TRAIN_CONFIG = config['stream'] | config['train']


def train():
    parser = argparse.ArgumentParser("Train CNN or ViT on the CIFAR-100 dataset.")
    
    parser.add_argument('--root'    , type=str  , default=TRAIN_CONFIG.pop('data_root') , help='The root directory of the dataset.')
    parser.add_argument('--output'  , type=str  , default=TRAIN_CONFIG.pop('output_dir'), help='The directory of the output files.')
    
    parser.add_argument('--batch'   , type=int  , default=TRAIN_CONFIG.pop('batch_size'), help='The size of a mini-batch in the DataLoader.')
    parser.add_argument('--model'   , type=str  , required=True                         , help='Model architecture, should be in ["CNN", "ViT"].')
    parser.add_argument("--epochs"  , type=int  , default=TRAIN_CONFIG.pop('epochs')    , help='Training epoch of the model.')    
    parser.add_argument('--fc_lr'   , type=float, default=TRAIN_CONFIG.pop('fc_lr')     , help='The learning rate of the fully-connect linear layer.')
    parser.add_argument("--ft_lr"   , type=float, default=TRAIN_CONFIG.pop('ft_lr')     , help='The fine-tuning learning rate of the model except the last fc layer.')
    
    parser.add_argument('--save'    , action='store_true'                               , help='Whether the model should be saved.')
    parser.add_argument('--beta'    , type=float, default=TRAIN_CONFIG.pop('beta')      , help='The beta hyper-params used in CutMix.')

    args = parser.parse_args()
    
    train_with_params(
        nn_name=args.model,
        epochs=args.epochs,
        ft_lr=args.ft_lr,
        fc_lr=args.fc_lr,
        save=args.save,
        data_root=args.root,
        output_dir=args.output,
        batch_size=args.batch,
        beta=args.beta,
        **TRAIN_CONFIG
    )
    
    
if __name__ == '__main__':
    train()

