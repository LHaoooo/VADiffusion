# writer ： Liuhao
# create_time ： 2023/5/11 11:20
# file_name：frame_pred_train.py

'''
Train a diffusion model to predict the target I frames
'''
import argparse
import copy
import traceback
import os,torch,xlwt
import time,datetime
import logging,yaml,sys,glob,shutil
import numpy as np
import torch.utils.tensorboard as tb

from utils.ncsn_runner import NCSNRunner

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, default='/home/VADiffusion/configs/UCSD_ped2_ddpm.yml',
                         required=True, help='Path to the config file')
    parser.add_argument('--data_path', type=str, default='/home/Dataset/UCSD_ped/UCSD_ped2',
                         required=True, help='The basic Path to the dataset')
    parser.add_argument('--seed', type=int, default=114514, help='Random seed')
    parser.add_argument('--device_ids', type=str, default='3,4,6,7', help='the ids of devices used')
    parser.add_argument('--exp', type=str, default='exp', required=True,
                         help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info',
                         help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--train', type=bool, default=True, help='Whether to train the model')
    parser.add_argument('--no_ema', action='store_true', help="Don't use Exponential Moving Average")
    parser.add_argument('--config_mod', nargs='*', type=str, default=[],
                        help='Overrid config options, e.g., model.ngf=64 model.spade=True training.batch_size=32')   # 这表示可以在命令行中去修改config中的参数

    args = parser.parse_args()

    args.command = 'python ' + ' '.join(sys.argv)  # 包含了当前脚本的命令行调用方式及其所有参数，可以进行保存
    args.log_path = os.path.join(args.exp, 'logs')

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Override with config_mod
    # 当命令行使用了--config时，覆盖config文件中的相关参数
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

    # Override
    # if config['data']['dataset'].upper() == "IMAGENET":
    #     if config['data']['classes'] is None:
    #         config['data']['classes'] = []
    #     elif config['data']['classes'] == 'dogs':
    #         config['data']['classes'] = list(range(151, 269))
    #     assert isinstance(config['data']['classes'], list), "config['data']['classes'] must be a list!"
    # config['sampling']['subsample'] = args.subsample or config['sampling'].get('subsample')  # 用args.subsample赋值，如果为None则该值不变
    # config['fast_fid']['batch_size'] = args.fid_batch_size or config['fast_fid']['batch_size']
    # config['fast_fid']['num_samples'] = args.fid_num_samples or config['fast_fid']['num_samples']
    # config['fast_fid']['pr_nn_k'] = args.pr_nn_k or config['fast_fid'].get('pr_nn_k', 3)

    if args.no_ema:
        config['model']['ema'] = False

    if config['sampling'].get('fvd', False) and config['sampling'].get('num_frames_pred', 10) < 10:
        print(" <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< WARNING: Cannot test FVD when sampling.num_frames_pred < 10 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        config['sampling']['fvd'] =  False

    if config['model'].get('output_all_frames', False): 
        noise_in_cond = True # if False, then wed predict the input-cond frames z, but the z is zero everywhere which is weird and seems irrelevant to predict. So we stick to the noise_in_cond case.

    assert not config['model'].get('cond_emb', False) or (config['model'].get('cond_emb', False) and config['data'].get('prob_mask_cond',0.0) > 0)

    if config['data'].get('prob_mask_sync', False):
        assert config['data'].get('prob_mask_cond', 0.0) > 0 and config['data'].get('prob_mask_cond', 0.0) == config['data'].get('prob_mask_future', 0.0)

    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    if args.train:  # train
        if not args.resume_training:  # 无预训练
            if os.path.exists(args.log_path):  # 是否重写log文件
                overwrite = False
                response = input(f"Folder {args.log_path} already exists.\nOverwrite? (Y/N)")
                if response.upper() == 'Y':
                    overwrite = True

                if overwrite:  # 覆盖
                    shutil.rmtree(args.log_path)  # 删除指定目录及其包含的所有文件和子目录。
                    # shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    # if os.path.exists(tb_path):
                    #     shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)  # 将Python字典config写入YAML文件的示例

            with open(os.path.join(args.log_path, 'args.yml'), 'w') as f:
                yaml.dump(vars(args), f, default_flow_style=False)  # vars(args)将args对象转换为一个字典

            # Code
            # code_path = os.path.join(args.exp, 'code')
            # os.makedirs(code_path, exist_ok=True)
            # copy_scripts(os.path.dirname(os.path.abspath(__file__)), code_path)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)  # 控制日志级别
        if not isinstance(level, int):  # ？？
            raise ValueError('level {} not supported'.format(args.verbose))

        # 配置两个处理器和一个格式化器，然后获取默认的日志器（logger）并为其添加这两个处理器。
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
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True

    # return args, new_config, config_uncond
    return args, new_config

'''
将Python字典类型的配置文件转换为命名空间(namespace)类型的对象。
具体来说,该函数将递归地遍历输入的字典类型config,
对于每个键值对(key, value),
将其转换为一个命名空间的属性，
并将其赋值给一个命名空间对象namespace的对应属性。
如果当前键对应的值仍然是一个字典，
则递归地调用dict2namespace函数,将其转换为一个子命名空间,
并将其赋值给当前命名空间对象的对应属性。

e.g.可以使用config.model.ngf来访问配置文件中的model.ngf选项的值。
'''
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# code copy
# def copy_scripts(src, dst):
#     print("Copying scripts in", src, "to", dst)
#     for file in glob.glob(os.path.join(src, '*.sh')) + \
#             glob.glob(os.path.join(src, '*.py')) + \
#             glob.glob(os.path.join(src, '*_means.pt')) + \
#             glob.glob(os.path.join(src, '*.data')) + \
#             glob.glob(os.path.join(src, '*.cfg')) + \
#             glob.glob(os.path.join(src, '*.yml')) + \
#             glob.glob(os.path.join(src, '*.names')):
#         shutil.copy(file, dst)
#     for d in glob.glob(os.path.join(src, '*/')):
#         if '__' not in os.path.basename(os.path.dirname(d)) and \
#                 '.' not in os.path.basename(os.path.dirname(d))[0] and \
#                 'ipynb' not in os.path.basename(os.path.dirname(d)) and \
#                 os.path.basename(os.path.dirname(d)) != 'data' and \
#                 os.path.basename(os.path.dirname(d)) != 'experiments' and \
#                 os.path.basename(os.path.dirname(d)) != 'assets':
#             if os.path.abspath(d) in os.path.abspath(dst):
#                 continue
#             print("Copying", d)
#             # shutil.copytree(d, os.path.join(dst, d))
#             new_dir = os.path.join(dst, os.path.basename(os.path.normpath(d)))
#             os.makedirs(new_dir, exist_ok=True)
#             copy_scripts(d, new_dir)

def main():
    args, config = parse_args_and_config()
    world_size = torch.cuda.device_count()
    config.trainging.batch_size *= world_size  # 多卡batchsize/lr要是单卡的多倍
    config.optim.lr *= world_size
    logging.info("{}".format(args))
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    # if not args.test and not args.sample and not args.fast_fid:
    #     del config_dict['tb_logger']
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
        # runner = NCSNRunner(args, config, config_uncond,
        #                     trainset_yuvroot,testset_yuvroot,trainset_mvroot,testset_mvroot)
        runner.train()
    except:
        logging.error(traceback.format_exc())
    
    logging.info(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == "__main__":
    main()