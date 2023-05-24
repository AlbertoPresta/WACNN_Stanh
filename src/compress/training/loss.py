
import torch 
import math
import torch.nn as nn

class RateDistortionLoss(nn.Module):

    def __init__(self, lmbda = 1e-2, hype = False):
        super().__init__()

        self.dist_metric = nn.MSELoss()
        self.lmbda = lmbda 
        self.hype = hype


    def forward(self, output, target):
        N, _, H, W = target.size()      
        out = {}
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)


        if self.hype:
            out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        else:
            out["z_bpp"]  = out["y_bpp"]

        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   
        
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"] 

        return out  