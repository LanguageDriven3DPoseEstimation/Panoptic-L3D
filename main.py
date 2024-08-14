import argparse

from configs.get_configs import get_cfg_default
from trainer import Trainer

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def update_cfg(args):
    cfg = get_cfg_default()
    cfg.set_new_allowed(True)
    cfg.merge_from_file('./configs/defaults.yaml')
    cfg.merge_from_file(f'./configs/stage/{args.training_stage}.yaml')

    cfg.RUNNING_MODE = args.running_mode
    cfg.PRETRAINED_PATH = args.pretrained_path
    cfg.TRAINING_STAGE = args.training_stage
    cfg.DATA.ROOT = args.dataset_root
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.NUM_WORKERS = args.num_workers
    cfg.TRAIN.NUM_EPOCHS = args.num_epochs
    cfg.OUTPUT_PATH = args.output_path
    cfg.LOAD_PRETRAINED_MODEL = args.load_pretrained_model

    cfg.freeze()

    return cfg

def main(args):
    config = update_cfg(args)
    print_args(args, config)

    trainer = Trainer(config)
    if args.running_mode == 'train':
        if args.training_stage == 'stage1':
            trainer.train_stage1()
        elif args.training_stage == 'stage2':
            trainer.train_stage2()
    elif args.running_mode == 'test':
        trainer.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CPP training and evaluation')
    parser.add_argument('--running_mode', '-rm', default='test', choices=['train', 'test'],
                        help="mode to run, either 'train' or 'eval'")
    parser.add_argument('--pretrained_path', type=str, default='./pretrained', help='path to pre-trained model')
    parser.add_argument('--training_stage',type=str,default='stage2',choices=['stage1', 'stage2'],help='stage to training')
    parser.add_argument('--dataset_root', type=str, default='/path/of/your/dataset_root/')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=12,help='number of workers for data')
    parser.add_argument('--output_path', type=str, default='/path/of/your/output/', help='output directory')
    parser.add_argument('--load_pretrained_model', type=bool, default=True, help='whether to load pretrained model')

    args = parser.parse_args()

    main(args)
