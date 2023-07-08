import numpy as np

import os
import torch
path = "stanh/"
path_models = "devil2022"
models_list = [os.path.join(path_models,l) for l in os.listdir(path_models)]
for q in models_list:
    print(q)
    checkpoint = torch.load(q, map_location= "cpu")
    weight = checkpoint[ "entropy_bottleneck_w"].detach().numpy()
    print(q,":",weight)
    qual = q.split("/")[-1].split("-")[0][-2:]
    
    np.save(path + qual + "zou22.npy",weight)