import os
import sys
import argparse
import logging
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="2",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        default=os.path.join(BASE_DIR, "config/REAL/camera_real.yaml"),
                        help="path to config file")
    parser.add_argument("--eval_gpu", type=str, default="3",
                        help="physical GPU used by asynchronous evaluation")
    parser.add_argument("--eval_interval", type=int, default=2,
                        help="run asynchronous test every N epochs; <=0 disables it")
    parser.add_argument("--eval_workers", type=int, default=4,
                        help="number of test dataloader workers")
    args_cfg = parser.parse_args()

    return args_cfg

def init():
    args = get_parser()
    if args.eval_interval > 0 and args.eval_gpu in {gpu.strip() for gpu in args.gpus.split(',')}:
        raise ValueError('--eval_gpu must be different from the training GPU(s)')
    # Select the physical training GPU before importing torch/CUDA extensions.
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
    import gorilla
    from solver import get_logger
    exp_name = args.config.split("/")[-1].split(".")[0]
    log_dir = os.path.join("log", exp_name)
    
    if not os.path.isdir("log"):
        os.makedirs("log")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.simeco = gorilla.Config.fromfile(os.path.join(BASE_DIR, 'config', 'simeco.yaml'))
    cfg.exp_name = exp_name
    cfg.log_dir = log_dir
    cfg.ckpt_dir = os.path.join(log_dir, 'ckpt')
    if not os.path.isdir(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)
        
    cfg.gpus = args.gpus
    cfg.config_path = os.path.abspath(args.config)
    cfg.eval_gpu = args.eval_gpu
    cfg.eval_interval = args.eval_interval
    cfg.eval_workers = args.eval_workers
    logger = get_logger(
        level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir+"/training_logger.log")
    return logger, cfg

if __name__ == "__main__":
    logger, cfg = init()
    import torch
    import gorilla
    from create_dataloaders import create_dataloaders
    from solver import Solver
    from Net import Net
    from model.losses import ComPoseProxyLoss as Loss

    logger.warning(
        "************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    # model
    logger.info("=> creating model ...")
    model = Net(cfg.pose_net, cfg.simeco)
    
    start_epoch = 1
    start_iter = 0
    
    model = model.cuda()
    # Create the lazy cuBLAS handle before activations occupy most GPU memory.
    torch.mm(torch.ones(1, 1, device='cuda'), torch.ones(1, 1, device='cuda'))
        
    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.warning("#Total parameters : {}".format(count_parameters))
    loss = Loss(
        cfg.loss,
        num_obj=cfg.simeco.num_query,
        axis_order='xyz',
        sym_ids=(0, 1, 3),
        axis_length_ratio=0.8,
        axis_diameter_nocs=0.12,
        yaw_bins=36
    ).cuda()
    
    # dataloader
    dataloaders = create_dataloaders(cfg.train_dataset)

    for k in dataloaders.keys():
        dataloaders[k].dataset.reset()

    # solver
    Trainer = Solver(model=model, 
                     loss=loss,
                     dataloaders=dataloaders,
                     logger=logger,
                     cfg=cfg,
                     start_epoch=start_epoch,
                     start_iter=start_iter)
    Trainer.solve()

    logger.info('\nFinish!\n')
