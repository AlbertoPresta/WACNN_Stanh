
import torch 
import torch.nn as nn
import random
import wandb
from compress.utils.help_function import compute_msssim, compute_psnr
from compressai.ops import compute_padding
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F

def read_image(filepath, clic =False):
    #assert filepath.is_file()
    img = Image.open(filepath)
    
    if clic:
        i =  img.size
        i = i[0]//2, i[1]//2
        img = img.resize(i)
    img = img.convert("RGB")
    return transforms.ToTensor()(img)
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


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm ,lmbdas , counter,stanh = False):
    model.train()
    device = next(model.parameters()).device



    
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()


    for i, d in enumerate(train_dataloader):
        counter += 1
        d = d.to(device)

        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()

        quality_index =  random.randint(0, len(lmbdas) - 1)
        lmbda_picked = lmbdas[quality_index]

        if stanh is False:
            out_net = model(d)
        else: 
            out_net = model(d, stanh_level = quality_index, training = True)





        out_criterion = criterion(out_net, d, lmbda_picked)
        out_criterion["loss"].backward()

        loss.update(out_criterion["loss"].clone().detach())
        mse_loss.update(out_criterion["mse_loss"].clone().detach())
        bpp_loss.update(out_criterion["bpp_loss"].clone().detach())


        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()


        if aux_optimizer is not None:
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

        if i % 10000 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                
                f'\tMSE loss: {out_criterion["mse_loss"].item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'

            )



        wand_dict = {
            "train_batch": counter,
            #"train_batch/delta": model.gaussian_conditional.sos.delta.data.item(),
            "train_batch/losses_batch": out_criterion["loss"].clone().detach().item(),
            "train_batch/bpp_batch": out_criterion["bpp_loss"].clone().detach().item(),
            "train_batch/mse":out_criterion["mse_loss"].clone().detach().item(),
        }
        wandb.log(wand_dict)

    log_dict = {
        "train":epoch,
        "train/loss": loss.avg,
        "train/bpp": bpp_loss.avg,
        "train/mse": mse_loss.avg,
        }
        
    wandb.log(log_dict)
    return counter


def test_epoch(epoch, test_dataloader, model, criterion,  lmbdas,valid, stanh = False):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()


    psnr = AverageMeter()
    ssim = AverageMeter()
    c = 0
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)

            if valid == False:
                quality_index =  random.randint(0, len(lmbdas) - 1)
            else: 
                quality_index = c%len(lmbdas)
            lmbda_picked = lmbdas[quality_index]

            if stanh is False:
                out_net = model(d)
            else:
                out_net = model(d, stanh_level = quality_index, training = False)

            out_criterion = criterion(out_net, d,lmbda_picked)

            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])




            psnr.update(compute_psnr(d, out_net["x_hat"]))
            ssim.update(compute_msssim(d, out_net["x_hat"]))



    if valid is False:
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
        )
        log_dict = {
        "test":epoch,
        "test/bpp":bpp_loss.avg,
        "test/psnr":psnr.avg,
        "test/ssim":ssim.avg,
        }
    else:

        print(
            f"valid epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
        )
        log_dict = {
        "valid":epoch,
        "valid/loss": loss.avg,
        "valid/bpp":bpp_loss.avg,
        "valid/mse": mse_loss.avg,
        "valid/psnr":psnr.avg,
        "valid/ssim":ssim.avg,
        }       

    wandb.log(log_dict)

    return loss.avg




def compress_with_ac(model, filelist, device, epoch, stanh = False, ql = 0):
    #model.update(None, device)
    print("ho finito l'update")
    bpp_loss = AverageMeter()
    psnr = AverageMeter()
    mssim = AverageMeter()

    
    with torch.no_grad():
        for i,d in enumerate(filelist): 
            print("-------------    ",i,"  --------------------------------")
            x = read_image(d).to(device)
            x = x.unsqueeze(0) 
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
            x_padded = F.pad(x, pad, mode="constant", value=0)

            if stanh is False:
                out_enc = model.compress(x_padded)
                out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            else:
                out_enc = model.compress(x_padded,stanh_level = ql)
                out_dec = model.decompress(out_enc["strings"], out_enc["shape"],stanh_level = ql)
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])



            num_pixels = x.size(0) * x.size(2) * x.size(3)
            bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
            out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
            out_dec["x_hat"].clamp_(0.,1.)

            bpp_loss.update(bpp)
            psnr.update(compute_psnr(x, out_dec["x_hat"]))
            mssim.update(compute_psnr(x, out_dec["x_hat"]))

                                 
    log_dict = {
            "compress":epoch,
            "compress/bpp": bpp_loss.avg,
            "compress/psnr": psnr.avg,
            "compress/mssim":mssim.avg
    }
    
    wandb.log(log_dict)
    return bpp_loss.avg



import shutil
def save_checkpoint(state, is_best, filename,filename_best):
    torch.save(state, filename)
    wandb.save(filename)
    if is_best:
        shutil.copyfile(filename, filename_best)
        wandb.save(filename_best)



from os.path import join         
def create_savepath(args):

    

    
    c_best = "best_"
    c = "last_"
    c = c + args.suffix
    c_best = c_best + args.suffix

    
    
    path = args.savepath
    savepath = join(path,c)
    savepath_best = join(path,c_best)
    
    print("savepath: ",savepath)
    print("savepath best: ",savepath_best)#ddd
    return savepath, savepath_best
