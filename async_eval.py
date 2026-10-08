import argparse
import logging
import os
import random
import sys
import tempfile


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='Asynchronous pose evaluation')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--epoch', required=True, type=int)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.path.extend([os.path.join(BASE_DIR, p) for p in
                     ('provider', 'model', 'model/pointnet2', 'utils')])

    import gorilla
    import torch
    from Net import Net
    from nocs_dataset import TestDataset
    from solver import get_logger, test_func
    from evaluation_utils import evaluate

    cfg = gorilla.Config.fromfile(args.config)
    cfg.simeco = gorilla.Config.fromfile(os.path.join(BASE_DIR, 'config', 'simeco.yaml'))
    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    log_dir = os.path.join(BASE_DIR, 'log', exp_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(logging.INFO, logging.WARNING,
                        os.path.join(log_dir, 'async_eval.log'),
                        name_logger=f'async_eval_epoch_{args.epoch}')
    logger.warning(f'===== asynchronous evaluation: epoch {args.epoch} =====')

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    model = Net(cfg.pose_net, cfg.simeco).cuda().eval()
    gorilla.solver.load_checkpoint(model=model, filename=args.checkpoint)
    dataset = TestDataset(cfg.test_dataset.img_size, cfg.test_dataset.sample_num,
                          cfg.test_dataset.dataset_dir, cfg.setting,
                          cfg.test_dataset.dataset_name)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers,
                                         shuffle=False, drop_last=False)
    with tempfile.TemporaryDirectory(prefix=f'eval_epoch_{args.epoch}_', dir=log_dir) as result_dir:
        test_func(model, loader, result_dir)
        evaluate(result_dir, logger, cat_id=-1)
    logger.warning(f'Epoch {args.epoch}: evaluation complete; temporary result files removed')


if __name__ == '__main__':
    main()
