import argparse
import math
import random
import sys
import os
import time
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_path, transform):
        self.data_dir = data_path
        self.dataset_list = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.dataset_list[idx])
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out['y_bpp'] = torch.log(output['likelihoods']['y']).sum() / (-math.log(2) * num_pixels)
        out['z_bpp'] = torch.log(output['likelihoods']['z']).sum() / (-math.log(2) * num_pixels)
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        out["psnr"] = 10 * (torch.log(1 * 1 / out["mse_loss"]) / np.log(10))

        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, epoch, global_step, clip_max_norm
):
    model.train()
    print(model.training)
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    t_start = time.time()
    for i, d in enumerate(train_dataloader):

        global_step+=1
        d = d.to(device)
        optimizer.zero_grad()
        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            if total_norm.isnan() or total_norm.isinf():
                print("non-finite norm, skip this batch")
                continue
        optimizer.step()

        bpp_loss.update(out_criterion["bpp_loss"])
        loss.update(out_criterion["loss"])
        mse_loss.update(out_criterion["mse_loss"])
        psnr.update(out_criterion["psnr"])
        y_bpp.update(out_criterion["y_bpp"])
        z_bpp.update(out_criterion["z_bpp"])

        if i % 100 == 0 :
            t_end = time.time()-t_start
            t_start = time.time()
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f"\tLoss: {loss.avg:.4f} |"
                f"\tMSE loss: {mse_loss.avg:.6f} |"
                f"\tPSNR: {psnr.avg:.3f} |"
                f"\tBpp loss: {bpp_loss.avg:.4f} |"
                f"\ty bpp: {y_bpp.avg:.4f} |"
                f"\tz bpp: {z_bpp.avg:.4f} |"
                f'\t time : {t_end:.2f} |'
            )
            torch.cuda.empty_cache()
        
    return global_step


def test_epoch(epoch, test_dataloader, model, criterion, writer):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion["psnr"])
            y_bpp.update(out_criterion["y_bpp"])
            z_bpp.update(out_criterion["z_bpp"])
    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.4f} |"
        f"\tMSE loss: {mse_loss.avg:.6f} |"
        f"\tPSNR: {psnr.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\ty bpp: {y_bpp.avg:.4f} |"
        f"\tz bpp: {z_bpp.avg:.4f} |"
    )
    writer.add_scalar("test_loss", loss.avg, global_step = epoch)
    writer.add_scalar("test_mse_loss", mse_loss.avg, global_step = epoch)
    writer.add_scalar("test_bpp_loss", bpp_loss.avg, global_step = epoch)

    return loss.avg

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_class", type=str, default="hypers")
    parser.add_argument(
        "-tr_d", "--train_dataset", type=str, help="Training dataset"
    )
    parser.add_argument(
        "-te_d", "--test_dataset", type=str, help="Testing dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=3001,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.013,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=32, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="/output/", help="Where to Save model"
    )
    parser.add_argument(
        "--log_dir", type=str, default="/output/", help="Where to Save logs"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    print(args)
    args.log_dir = os.path.join(args.log_dir, args.model_name + '_lmbda' + str(args.lmbda))
    args.save_path = os.path.join(args.save_path, args.model_name + '_lmbda' + str(args.lmbda))
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    test_dataset = Dataset(
        args.test_dataset,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    train_dataset = Dataset(
        args.train_dataset,
        transform=transforms.Compose([
            transforms.RandomCrop(args.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    import importlib
    net = importlib.import_module(f'.{args.model_name}', f'src.models').HPCM()
    print(net)
    net = net.to(device)

    lr_scheduler = lambda x : \
    1e-4 if x < 2750 else (
        3e-5 if x < 2850 else (
            1e-5 if x < 2950 else 1e-6
        )
    )

    last_epoch = 0

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    writer = SummaryWriter(args.log_dir)

    best_loss = float("inf")
    global_step = 0
    for epoch in range(last_epoch, args.epochs):

        lr = lr_scheduler(epoch)
        for param_group in optimizer.param_groups: 
            param_group['lr'] = lr
        
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        global_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            global_step,
            args.clip_max_norm,
        )

        loss = test_epoch(epoch, test_dataloader, net, criterion, writer)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if is_best:
            print(f"epoch {epoch} is best now!")
            torch.save(net.state_dict(), os.path.join(args.save_path, 'epoch_' +'best' + '.pth.tar'))

        if epoch % 1000 == 0:
            torch.save(net.state_dict(), os.path.join(args.save_path, 'epoch_' + str(epoch) + '.pth.tar'))


if __name__ == "__main__":
    main(sys.argv[1:])
