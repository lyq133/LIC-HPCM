import math
import glob
import time
import torch
import argparse
import numpy as np
from PIL import Image
from typing import Dict, Any
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from pytorch_msssim import ms_ssim
       
def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )
    
def load_image(filepath: str):
    return Image.open(filepath).convert("RGB")

def img2torch(img: Image.Image):
    return ToTensor()(img).unsqueeze(0)

def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255):
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())

def compute_metrics(
    org: torch.Tensor, rec: torch.Tensor, max_val: int = 255):
    metrics: Dict[str, Any] = {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    return metrics

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        if type(val) == torch.Tensor:
            val = val.detach().cpu()

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def get_scale_table(min, max, levels):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def test(args):
    device = torch.device("cuda")
    ##### dataset
    images_list = glob.glob(f'{args.dataset}/*.png')

    ##### load model
    import importlib
    net = importlib.import_module(f'.{args.model_name}', f'src.models').HPCM
        
    args.checkpoint = [args.checkpoint]
    # suggest:
    # args.checkpoint = [
    #     '/path-to-ckpt/0.0018.pth.tar', 
    #     '/path-to-ckpt/0.0035.pth.tar', 
    #     '/path-to-ckpt/0.0067.pth.tar', 
    #     '/path-to-ckpt/0.013.pth.tar', 
    #     '/path-to-ckpt/0.025.pth.tar', 
    #     '/path-to-ckpt/0.0483.pth.tar', 
    # ]
    bpp_all = []
    psnr_all = []
    ssim_all = []
    for ckpt in args.checkpoint:
        print("Loading", ckpt)
        checkpoint = torch.load(ckpt, map_location=device)
        model = net()
        model.eval()
        model.load_state_dict(checkpoint, strict=True)
        model.update(get_scale_table(0.12, 64, args.num))
        model = model.to(device)

        bpp_loss = AverageMeter()
        psnr = AverageMeter()
        ssim = AverageMeter()
        y_bpp = AverageMeter()
        z_bpp = AverageMeter()
        enc_time = AverageMeter()
        dec_time = AverageMeter()

        for img_path in sorted(images_list):
            
            img = load_image(img_path)
            x = img2torch(img)
            h, w = x.size(2), x.size(3)
            x = x.to(device)
            p = 256
            x_pad = pad(x, p)
            img_name = img_path.split('/')[-1]
            print(img_name)
            torch.cuda.synchronize()
            enc_start = time.time()
            with torch.no_grad():
                out_enc = model.compress(x_pad)
            torch.cuda.synchronize()
            enc_t = time.time() - enc_start
            
            torch.cuda.synchronize()
            dec_start = time.time()
            with torch.no_grad():
                out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            torch.cuda.synchronize()
            dec_t = time.time() - dec_start
            x_hat = crop(out_dec["x_hat"], (h,w))

            psnr_img = compute_metrics(x, x_hat, 255)['psnr']

            msssim = ms_ssim(x_hat, x, data_range=1.0)
            msssim_db = 10 * (torch.log(1 * 1 / (1 - msssim)) / np.log(10)).item()

            num_pixels = h*w
            bpp_img = sum(len(s) for s in out_enc["strings"]) * 8.0 / num_pixels
            ybpp_img = len(out_enc["strings"][0]) * 8.0 / num_pixels
            zbpp_img = len(out_enc["strings"][1]) * 8.0 / num_pixels

            print('image name:',img_name)
            print(
                f"{img_name}"
                f"\tPSNR: {psnr_img} |"
                f"\tMS-SSIM: {msssim_db} |"
                f"\tBpp loss: {bpp_img} |"
                f"\ty bpp: {ybpp_img} |"
                f"\tz bpp: {zbpp_img} |"
                f"\tenc time: {enc_t} |"
                f"\tdec time: {dec_t} |"
            )

            bpp_loss.update(bpp_img)
            psnr.update(psnr_img)
            ssim.update(msssim_db)
            y_bpp.update(ybpp_img)
            z_bpp.update(zbpp_img)
            enc_time.update(enc_t)
            dec_time.update(dec_t)
        print(
            f"Test:"
            f"\tPSNR: {psnr.avg} |"
            f"\tMS-SSIM: {ssim.avg} |"
            f"\tBpp loss: {bpp_loss.avg} |"
            f"\ty bpp: {y_bpp.avg} |"
            f"\tz bpp: {z_bpp.avg} |"
            f"\tenc time: {enc_time.avg} |"
            f"\tdec time: {dec_time.avg} |"
        )
        bpp_all.append(bpp_loss.avg)
        psnr_all.append(psnr.avg)
        ssim_all.append(ssim.avg)
    print(bpp_all)
    print(psnr_all)
    print(ssim_all)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--model_name", type=str, default="HPCM_Base")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("-num", "--num", type=int, default=60)
    parser.add_argument("-data", "--dataset", type=str, default='')
    args = parser.parse_args()
    print(args)
    test(args)
