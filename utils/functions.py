import torch
import logging
import torch.nn as nn
import numpy as np
import math
from .pytorch_ssim import *
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count


def linear_warm_up_lr(optimizer, epoch, warm_up_steps, lr):
    for param_group in optimizer.param_groups:
        warm_lr = lr*(epoch+1.0)/warm_up_steps
        param_group['lr'] = warm_lr


def get_logger(filename):
    logger = logging.getLogger('train_logger')

    while logger.handlers:
        logger.handlers.pop()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, 'w')
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s], ## %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def val(model, test_loader, factor=1.0, verbose=False, downsample = 8):
    print('validation on whole images!')
    model.eval()
    mae, rmse = 0.0, 0.0
    psnr = 0.0
    ssim = 0.0
    ssim_d = 0.0
    psnr_d = 0.0
    with torch.no_grad():
        for it,data in enumerate(test_loader):
            img, target, count = data[0:3]
            
            img = img.cuda()
            target = target.unsqueeze(1).cuda()
            output = model(img)
            if isinstance(output, tuple):
                dmp, amp = output
                hard_amp = (amp > 0.5).float()
                dmp = dmp * hard_amp
            else:
                dmp = output
            est_count = dmp.sum().item()/factor
            if verbose:
                print('gt:{:.1f}, est:{:.1f}'.format(count.item(),est_count))
            elif it < 10:
                print('gt:{:.1f}, est:{:.1f}'.format(count.item(),est_count))
            mae += abs(est_count - count.item())
            rmse += (est_count - count.item())**2
            if verbose:
                mse = torch.mean((dmp - target)**2).float()
                
                psnr += 10 * math.log(1.0/mse, 10)
                ssim += cal_ssim(dmp, target)
                
                dmp_d = F.interpolate(dmp, [target.shape[2]//downsample, target.shape[3]//downsample]) * downsample**2
                target_d = F.interpolate(target, [target.shape[2]//downsample, target.shape[3]//downsample]) * downsample**2
                mse_d = torch.mean((dmp_d - target_d)**2).float()
                psnr_d += 10 * math.log(1.0/mse_d, 10)
                ssim_d += cal_ssim(dmp_d, target_d)
    mae /= len(test_loader)
    rmse /= len(test_loader)
    rmse = rmse**0.5
    psnr /= len(test_loader)
    ssim /= len(test_loader)
    psnr_d /= len(test_loader)
    ssim_d /= len(test_loader)
    if verbose:
        print('psnr:{:.2f}, ssim:{:.4f}, psnr of 1/8 size:{:.2f}, ssim of 1/8 size:{:.2f}'.format(psnr, ssim, psnr_d, ssim_d))
    return mae, rmse


def test_ssim():
    img1 = (torch.rand(1, 1, 16, 16))
    img2 = (torch.rand(1, 1, 16, 16))

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()
    print(torch.max(img1))
    print(torch.max(img2))
    print(max(torch.max(img1),torch.max(img2)))
    print(cal_ssim(img1, img2).float())
    
    
    
# validate on bayes dataloader
def val_bayes(model, test_loader, factor=1.0, verbose=False):
    print('validation on bayes loader!')
    model.eval()
    epoch_res=[]
    for it,(inputs, count, name) in enumerate(test_loader):
        inputs = inputs.cuda()
        # inputs are images with different sizes
        assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            est = torch.sum(outputs).item()/factor
            res = count[0].item() - est
            if verbose:
                print('gt:{:.1f}, est:{:.1f}'.format(count[0].item(),torch.sum(outputs).item()))
            elif it<10:
                print('gt:{:.1f}, est:{:.1f}'.format(count[0].item(),torch.sum(outputs).item()))
            epoch_res.append(res)
    epoch_res = np.array(epoch_res)
    rmse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))
    return mae, rmse


# validate with 4 non-overlapping patches
def val_patch(model, test_loader, factor=1.0, verbose=False):
    print('validaiton on 4 quarters!')
    model.eval()
    mae, rmse = 0.0, 0.0
    with torch.no_grad():
        for it, data in enumerate(test_loader):
            img, _, count = data[0:3]
            h,w = img.shape[2:]
            h_d = h//2
            w_d = w//2
            
            img_1 = (img[:,:,:h_d,:w_d].cuda())
            img_2 = (img[:,:,:h_d,w_d:].cuda())
            img_3 = (img[:,:,h_d:,:w_d].cuda())
            img_4 = (img[:,:,h_d:,w_d:].cuda())
            img_patches = [img_1, img_2, img_3, img_4]
            est_count = 0
            for img_p in img_patches:
                output = model(img_p)
                if isinstance(output, tuple):
                    dmp, amp = output
                    dmp = dmp *amp
                else:
                    dmp = output
                est_count += dmp.sum().item()/factor
            if verbose:
                print('gt:{:.1f}, est:{:.1f}'.format(count.item(),est_count))
            elif it < 10:
                print('gt:{:.1f}, est:{:.1f}'.format(count.item(),est_count))
            mae += abs(est_count - count.item())
            rmse += (est_count - count.item())**2
    mae /= len(test_loader)
    rmse /= len(test_loader)
    rmse = rmse**0.5
    return mae, rmse