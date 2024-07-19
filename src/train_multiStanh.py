import torch 
import os 
import numpy as np 
from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import math
from compressai.ops import compute_padding
import math 
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
import numpy as np
import sys

import argparse
from compressai.zoo import *
from torch.utils.data import DataLoader
from os.path import join 
from compress.zoo import models, aux_net_models
import wandb
from torch.utils.data import Dataset
from os import listdir
from collections import OrderedDict


torch.backends.cudnn.benchmark = True

def update_checkpopoint(state_dict,num_stanh):

    res =  OrderedDict()


    for k,v in state_dict.items():
        if "gaussian_conditional" in k:
            for j in range(num_stanh):
                adding = str(j) 
                new_text = k.replace("gaussian_conditional.", "gaussian_conditional." + adding + ".")
                res[new_text] = state_dict[k]
        elif "entropy_bottleneck" in k:
            for j in range(num_stanh):
                adding = str(j) 
                new_text = k.replace("entropy_bottleneck.", "entropy_bottleneck." + adding + ".")
                res[new_text] = state_dict[k]
        else:
            res[k]=state_dict[k]
    
    return res

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

image_models = {"zou22-base": aux_net_models["stf"],
                "zou22-sos":models["cnn_multi"],

                }




def rename_key(key):
    """Rename state_deeict key."""

    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]
    if key.startswith('h_s.'):
        return None

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    # if ".downsample." in key:
    #     return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key



def load_state_dict(state_dict):
    """Convert state_dict keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    if None in state_dict:
        state_dict.pop(None)
    return state_dict

def load_checkpoint(arch: str, checkpoint_path: str):
    state_dict = load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return models[arch].from_state_dict(state_dict).eval()

class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = [os.path.join(self.data_dir,f) for f in os.listdir(self.data_dir)]

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose(
        [transforms.CenterCrop(256), transforms.ToTensor()]
    )
        return transform(image)

    def __len__(self):
        return len(self.image_path)


def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("-m","--model",default="3anchorsbis",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-mp","--model_path",default="/scratch/inference/new_models/devil2022/",help="Model architecture (default: %(default)s)",)
    
    
    parser.add_argument("-sp","--stanh_path",default="/scratch/inference/new_models/devil2022/3_anchors_stanh",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-rp","--result_path",default="/scratch/inference/results",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-ip","--image_path",default="/scratch/dataset/kodak",help="Model architecture (default: %(default)s)",)
    parser.add_argument("--entropy_estimation", action="store_true", help="Use cuda")


    args = parser.parse_args(argv)
    return args


def bpp_calculation(out_net, out_enc):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]

        bpp_1 = (len(out_enc[0]) * 8.0 ) / num_pixels
        #print("la lunghezza è: ",len(out_enc[1]))
        bpp_2 =  sum( (len(out_enc[1][i]) * 8.0 ) / num_pixels for i in range(len(out_enc[1])))
        return bpp_1 + bpp_2, bpp_1, bpp_2


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics( org, rec, max_val: int = 255):
    metrics =  {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics


def read_image(filepath):
    #assert filepath.is_file()
    img = Image.open(filepath)
    img = img.convert("RGB")
    return transforms.ToTensor()(img)


def evaluation(model,filelist,entropy_estimation,device):



    levels = [i for i in range(model.num_stanh)]

    psnr = [AverageMeter() for _ in range(model.num_stanh)]
    ms_ssim = [AverageMeter() for _ in range(model.num_stanh)]
    bpps =[AverageMeter() for _ in range(model.num_stanh)]



    bpp_across, psnr_across = [],[]
    for j in levels:
        print("***************************** ",j," ***********************************")
        for i,d in enumerate(filelist):
            name = "image_" + str(i)
            print(name," ",d," ",i)

            x = read_image(d).to(device)
            x = x.unsqueeze(0) 
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
            x_padded = F.pad(x, pad, mode="constant", value=0)

            

            if entropy_estimation is False: #ddd
                #print("entro qua!!!!")
                data =  model.compress(x_padded)
                out_dec = model.decompress(data)

            else:
                with torch.no_grad():
                    #print("try to do ",d)
                    out_dec = model(x_padded, training = False, stanh_level = j)
                    #print("done, ",d)
            if entropy_estimation is False:
                out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
                out_dec["x_hat"].clamp_(0.,1.)
                metrics = compute_metrics(x, out_dec["x_hat"], 255)
                size = out_dec['x_hat'].size()
                num_pixels = size[0] * size[2] * size[3]

                bpp ,_, _= bpp_calculation(out_dec, data["strings"]) #ddd

                
                metrics = compute_metrics(x_padded, out_dec["x_hat"], 255)
                print("fine immagine: ",bpp," ",metrics)

            else:
                out_dec["x_hat"].clamp_(0.,1.)
                out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
                size = out_dec['x_hat'].size()
                num_pixels = size[0] * size[2] * size[3]
                bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_dec["likelihoods"].values())
                metrics = compute_metrics(x, out_dec["x_hat"], 255)
                #print("fine immagine: ",bpp," ",metrics)
            

            
            bpps[j].update(bpp)
            psnr[j].update(metrics["psnr"]) #fff

            clear_memory()
        
        bpp_across.append(bpps[j].avg)
        psnr_across.append(psnr[j].avg)
    

    print(bpp_across," RISULTATI ",psnr_across)
    return bpp_across, psnr_across


def clear_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()




def main(argv):
    set_seed()
    args = parse_args(argv)
    model_name = args.model  # nome del modello che voglio studiare (ad esempio cheng2020)
    models_path = join(args.model_path,model_name) # percorso completo per arrivare ai modelli salvati (/scratch/inference/pretrained_models/chegn2020) qua ho salvato i modelli 
    device = "cuda"

    model_checkpoint =models_path + "/q5-zou22.pth.tar" # this is the 
    checkpoint = torch.load(model_checkpoint, map_location=device)

    checkpoint["state_dict"]["gaussian_conditional._cdf_length"] = checkpoint["state_dict"]["gaussian_conditional._cdf_length"].ravel()
    factorized_configuration =checkpoint["factorized_configuration"]
    factorized_configuration["trainable"] = True
    gaussian_configuration =  checkpoint["gaussian_configuration"]
    gaussian_configuration["trainable"] = True

    stanh_checkpoints_p = [args.stanh_path + "/q5-stanh.pth.tar",args.stanh_path + "/q4-stanh.pth.tar",
                         args.stanh_path + "/q3-stanh.pth.tar"]
    

    stanh_checkpoints = []

    for p in stanh_checkpoints_p:
        stanh_checkpoints.append(torch.load(p, map_location=device))





    #define model 
    architecture =  models["cnn_multi"]


    
    model =architecture(N = 192, 
                            M = 320, 
                            num_stanh = 3,
                            factorized_configuration = factorized_configuration, 
                            gaussian_configuration = gaussian_configuration)
            

    model = model.to(device)
    model.update()


    checkpoint["state_dict"] = update_checkpopoint(checkpoint["state_dict"],num_stanh = 3)
    model.load_state_dict(checkpoint["state_dict"],stanh_checkpoints)




       
    entropy_estimation = args.entropy_estimation
    
    images_path = args.image_path # path del test set 
    #savepath = args.result_path # path dove salvare i risultati 
    image_list = [os.path.join(images_path,f) for f in listdir(images_path)]
    
    model.freeze_net()
    bpp, psnr = evaluation(model,image_list,entropy_estimation,device)


if __name__ == "__main__":

    wandb.init(project="prova", entity="albipresta")   
    main(sys.argv[1:])