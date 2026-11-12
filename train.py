import torch
import argparse
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models.dataset import mesh_pc_dataset
from models.score_match import ScoreMatchNet
from utils.misc import *


# Arguments
parser = argparse.ArgumentParser()

# Dataset and loader
parser.add_argument('--data_path', type=str, default='data/train/npz')
parser.add_argument('--num_sample', type=int, default=1024)
parser.add_argument('--train_batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=1)

# Model architecture
parser.add_argument('--num_clean_nbs', type=int, default=10)
parser.add_argument('--dsm_sigma', type=float, default=0.01)
parser.add_argument('--score_net_hidden_dim', type=int, default=256)
parser.add_argument('--score_net_num_blocks', type=int, default=8)

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=float("inf"))

# Training
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs')
parser.add_argument('--dataset', type=str, default='PUNet')
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=1*MILLION)
parser.add_argument('--save_freq', type=int, default=20000)
args = parser.parse_args()
seed_all(args.seed)


# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='D%s_' % (args.dataset), postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, log_dir, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)


# Datasets and loaders
logger.info('Loading datasets')
train_dset = mesh_pc_dataset(args.data_path, mode='train', num_sample=args.num_sample)
train_iter = get_data_iterator(DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True))


# Model
logger.info('Building model...')
model = ScoreMatchNet(args).to(args.device)
logger.info(repr(model))


# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)


# Train
def train(it):
    # Load data
    batch = next(train_iter)
    sample_pc = batch['sample_pc'].to(args.device)  # (2, 1024, 3)
    points_gt = batch['points_gt'].to(args.device)  # (2, 10000, 3)
    closest_points = batch['closest_points'].to(args.device)    # (2, 1024, 3)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model.get_loss(sample_pc=sample_pc, points_gt=points_gt, closest_points=closest_points)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    # Logging
    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.6f' % (
        it, loss.item(), orig_grad_norm,
    ))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush() 


# Main loop
logger.info('Start training...')
try:
    for it in range(1, args.max_iters+1):
        train(it)
        if it % args.save_freq == 0:
            opt_states = {
                'optimizer': optimizer.state_dict(),
            }
            ckpt_mgr.save(model, args, opt_states, step=it)

except KeyboardInterrupt:
    print()
    logger.info('Terminating...')
