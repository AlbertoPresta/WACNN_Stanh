
import math
import random
import shutil
import sys
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from compress.training.loss import RateDistortionLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import os 
import glob
import argparse
from classic_functions import train_one_epoch, test_epoch, compress_with_ac, create_savepath, save_checkpoint
from compress.datasets import ImageFolder
from compress.zoo import models
from compress.utils.annealings import *
from compress.utils.help_function import CustomDataParallel, configure_optimizers, sec_to_hours
from torch.utils.data import Dataset
from PIL import Image
import time
def from_state_dict(cls, state_dict):

    net = cls(192, 320)
    net.load_state_dict(state_dict)
    return net
class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = [os.path.join(self.data_dir,f) for f in os.listdir(self.data_dir)]

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        #transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(image)

    def __len__(self):
        return len(self.image_path)


class RateDistortionLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.dist_metric = nn.MSELoss()


    def forward(self, output, target, lmbda):
        N, _, H, W = target.size()      
        out = {}
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)
        distortion = 255**2 * out["mse_loss"]
        num_pixels = N * H * W
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   
        out["loss"] = lmbda * distortion + out["bpp_loss"] 
        return out  


    def __len__(self):
        return len(self.image_path)
    


def rename_key(key):
    """Rename state_dict key.rrr"""

    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]
    if key.startswith('h_s.'):
        return None

    # ResidualBlockWithStride: 'downsample' -> 'skip'dd
    # if ".downsample." in key:
    #     return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters  pppp
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-m","--model",default="cnn_base",choices=models.keys(),help="Model architecture (default: %(default)s)",)
    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-e","--epochs",default=100,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)
    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)

    parser.add_argument("--lmbdas", nargs='+', type=float, default = [0.025]) #[ 0.0018,0.0035,0.0067,0.013,0.025,0.048]
    parser.add_argument("--sampling", action="store_true", help="Use cuda")



    parser.add_argument("--suffix",default=".pth.tar",type=str,help="factorized_annealing",)

    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size",type=int,default=64,help="Test batch size (default: %(default)s)",)
    parser.add_argument( "--aux-learning-rate", default=1e-3, type=float, help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--save", action="store_true", help="Save model to disk")
    parser.add_argument("--savepath", type=str, default="/scratch/StanH/mixed/", help="Where to Save model")
    parser.add_argument("--seed", type=float,default = 42, help="Set random seed for reproducibility")
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint",default = "/scratch/base_devil/weights/q6/model.pth")

    parser.add_argument("-ni","--num_images",default = 300000, type = int)
    parser.add_argument("-niv","--num_images_val",default = 1024, type = int)


    parser.add_argument("-dims","--dimension",default=192,type=int,help="Number of epochs (default: %(default)s)",) 
    parser.add_argument("-dims_m","--dimension_m",default=320,type=int,help="Number of epochs (default: %(default)s)",)

    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)


    if args.sampling:
        wandb.init(project="StanH_samp", config= args,entity="alberto-presta") 
    else:
        wandb.init(project="StanH_multlambda", config= args, entity="alberto-presta") 
    print(args,"cc")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )


    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms, num_images=args.num_images)
    valid_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms, num_images=args.num_images_val)
    test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak")
    device = "cuda" 

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,

    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,

    )


    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )


    N = args.dimension
    M = args.dimension_m
    net = models[args.model](N = N, M = M)



    if args.checkpoint != "none":  # load from previous checkpoint
        print("Loading the net", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        net.load_state_dict(checkpoint)



    net = net.to(device)



    if torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)


    lmbdas = args.lmbdas
    criterion = RateDistortionLoss()
    

    last_epoch = 0



    optimizer, aux_optimizer = configure_optimizers(net, args)
    print("hola!")
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=5)

    counter = 0
    best_loss = float("inf")
    epoch_enc = 0


    previous_lr = optimizer.param_groups[0]['lr']
    print("subito i paramteri dovrebbero essere giusti!")
    model_tr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    model_fr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad== False)
        
    print(" Ttrainable parameters: ",model_tr_parameters)
    print(" freeze parameters: ", model_fr_parameters)



    model_tr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    model_fr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad== False)
    print("******************************* DOPO")
    print(" trainable parameters: ",model_tr_parameters)
    print(" freeze parameters: ", model_fr_parameters)


    for epoch in range(last_epoch, args.epochs):
        print("**************** epoch: ",epoch,". Counter: ",counter)
        previous_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}","    ",previous_lr)
        print("epoch ",epoch)
        start = time.time()
        counter = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            lmbdas,
            counter,
        )


        loss_valid = test_epoch(epoch, valid_dataloader, net, criterion, lmbdas, valid = True)
        loss = test_epoch(epoch, test_dataloader, net, criterion, lmbdas, valid = False)

        lr_scheduler.step(loss_valid)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        filename, filename_best =  create_savepath(args)

        if  (is_best or epoch%25==0):
            save_checkpoint(
                    {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args":args
                },
                is_best,
                filename,
                filename_best
                )
            

        if epoch%5==0 or is_best:
            filelist = [os.path.join("/scratch/dataset/kodak", f) for f in os.listdir("/scratch/dataset/kodak")]
            print("entro qua")
            net.update()
            epoch_enc += 1
            compress_with_ac(net, filelist, device, epoch_enc) 



        print("log also the current leraning rate")

        log_dict = {
        "train":epoch,
        "train/leaning_rate": optimizer.param_groups[0]['lr'],
        #"train/beta": annealing_strategy_gaussian.beta
        }

        wandb.log(log_dict)
        end = time.time()
        print("Runtime of the epoch  ", epoch)
        sec_to_hours(end - start) 
        print("END OF EPOCH ", epoch)






if __name__ == "__main__":
      
    main(sys.argv[1:])
