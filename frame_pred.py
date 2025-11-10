# writer ： Liuhao
# create_time ： 2023/5/11 11:20
# file_name：frame_pred.py

'''
Train a diffusion model to predict the target I frames
'''
import argparse
import copy
import random
import traceback
import os,torch,xlwt
import time,datetime
import logging,yaml,sys,glob,shutil
import numpy as np
# import torch.utils.tensorboard as tb

from utils.ncsn_runner import NCSNRunner

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, default='/home/VADiffusion/configs/Shanghaitech_ddpm.yml',  # Avenue_  UCSD_ped2_ Shanghaitech_
                        help='Path to the config file')
    parser.add_argument('--data_path', type=str, default='/home/Dataset/shanghaitech',  # Avenue UCSD_ped/UCSD_ped2 shanghaitech
                        help='The basic Path to the dataset')
    parser.add_argument('--seed', type=int, default=123456, help='Random seed')
    parser.add_argument('--device_ids', type=str, default='0,1,2,3,4,5,6,7', help='the ids of devices used')
    parser.add_argument('--exp', type=str, default='exp1',
                         help='Path for saving running related data.Change the name to the different exp')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info',
                         help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--train', action='store_true', help='Whether to train the model')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--singletest', action='store_true', help='Whether to test the only model')
    parser.add_argument('--no_ema', action='store_true', help="Don't use Exponential Moving Average")
    parser.add_argument('--config_mod', nargs='*', type=str, default=[],
                        help='Overrid config options, e.g., model.ngf=64 model.spade=True training.batch_size=32')  

    args = parser.parse_args()

    args.command = 'python ' + ' '.join(sys.argv) 
    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # args.log_path = os.path.join(args.exp, 'debug')
    args.log_path = os.path.join(args.exp, 'logs_shanghai_test73_DDPM100')
    # Override with config_mod
    for val in args.config_mod:
        val, config_val = val.split('=')
        config_type, config_name = val.split('.')
        try:
            totest = config[config_type][config_name][0]
        except:
            totest = config[config_type][config_name]

        if isinstance(totest, str):
            config[config_type][config_name] = config_val
        else:
            config[config_type][config_name] = eval(config_val)

    if args.no_ema:
        config['model']['ema'] = False

    if config['sampling'].get('fvd', False) and config['sampling'].get('num_frames_pred', 10) < 10:
        print(" <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< WARNING: Cannot test FVD when sampling.num_frames_pred < 10 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        config['sampling']['fvd'] =  False

    assert not config['model'].get('cond_emb', False) or (config['model'].get('cond_emb', False) and config['data'].get('prob_mask_cond',0.0) > 0)

    if config['data'].get('prob_mask_sync', False):
        assert config['data'].get('prob_mask_cond', 0.0) > 0 and config['data'].get('prob_mask_cond', 0.0) == config['data'].get('prob_mask_future', 0.0)

    new_config = dict2namespace(config)

    if args.train:  # train
        if not args.resume_training:  
            if os.path.exists(args.log_path):  
                overwrite = False
                response = input(f"Folder {args.log_path} already exists.\nOverwrite? (Y/N)")
                if response.upper() == 'Y':
                    overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)  
                    os.makedirs(args.log_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False) 

            with open(os.path.join(args.log_path, 'args.yml'), 'w') as f:
                yaml.dump(vars(args), f, default_flow_style=False)  

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)  
        if not isinstance(level, int): 
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)


    # add device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    if torch.cuda.is_available() is False:
        raise EnvironmentError('not find GPU device for training.')
    
    device = torch.device('cuda')
    logging.info("Using device: {}{}".format(device, args.device_ids))
    new_config.device = device

    # config_uncond = new_config

    # set random seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True 

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    args, config = parse_args_and_config()
    world_size = torch.cuda.device_count()
    config.training.batch_size *= world_size 
    config.optim.lr *= world_size
    logging.info("{}".format(args))
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)
    logging.info("Args =")
    print(">" * 80)
    args_dict = copy.copy(vars(args))
    print(yaml.dump(args_dict, default_flow_style=False))
    print("<" * 80)

    trainset_yuvroot=os.path.join(args.data_path, "train_recyuv400/")
    testset_yuvroot=os.path.join(args.data_path,  "test_recyuv400/") 

    trainset_mvroot=os.path.join(args.data_path,  "trainmv_txt/")
    testset_mvroot=os.path.join(args.data_path,  "testmv_txt/")

    try:
        runner = NCSNRunner(args, config,
                            trainset_yuvroot,testset_yuvroot,trainset_mvroot,testset_mvroot)
        if args.train:
            runner.train()
        elif args.test:
            runner.test()
        elif args.singletest:
            runner.Single_test()
    except:
        logging.error(traceback.format_exc())
    
    logging.info(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == "__main__":
    main()