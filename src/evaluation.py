


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
                "zou22-sos":models["cnn"]
                }




from torch import Tensor

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



@torch.no_grad()
def test_epoch( test_dataloader, model,  sos):
    model.eval()
    device = next(model.parameters()).device
    bpp_loss = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()
    with torch.no_grad():
        for i,d in enumerate(test_dataloader):
            d = d.to(device)
            if sos:
                out_net = model(d, training = False)
            else: 
                out_net = model(d)


            N, _, H, W = out_net["x_hat"].size() 
            num_pixels = N*W*H
            bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_net["likelihoods"].values())
            bpp_loss.update(bpp)
            psnr.update(compute_psnr(d, out_net["x_hat"]))
            ssim.update(compute_msssim(d, out_net["x_hat"]))
            print("IMMAGINE ",i,"_ ",bpp,"-",compute_psnr(d, out_net["x_hat"]),"-",compute_msssim(d, out_net["x_hat"]))





    return  psnr.avg, ssim.avg, bpp_loss.avg




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


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("-m","--model",default="6anchors",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-mp","--model_path",default="/scratch/inference/new_models/devil2022/",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-rp","--result_path",default="/scratch/inference/results",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-ip","--image_path",default="/scratch/dataset/kodak",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-ep","--entropy_estimation",default=False,help="Model architecture (default: %(default)s)",)


    args = parser.parse_args(argv)
    return args

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)




def reconstruct_image_with_nn(networks, filepath, device, save_path):
    reconstruction = {}
    for name, net in networks.items():
        #net.eval()
        with torch.no_grad():
            x = read_image(filepath).to(device)
            x = x.unsqueeze(0)
            out_net,= net(x,  False)
            out_net["x_hat"].clamp_(0.,1.)
            original_image = transforms.ToPILImage()(x.squeeze())
            reconstruction[name] = transforms.ToPILImage()(out_net['x_hat'].squeeze())




    svpt = os.path.joint(save_path,"original" + filepath.split("/")[-1])

    fix, axes = plt.subplots(1, 1)
    for ax in axes.ravel():
        ax.axis("off")

    axes.ravel()[0 ].imshow(original_image)
    axes.ravel()[0].title.set_text("original image")    
    
    plt.savefig(svpt)
    plt.close()


    svpt = os.path.joint(save_path,filepath.split("/")[-1])

    fix, axes = plt.subplots(5, 4, figsize=(10, 10))
    for ax in axes.ravel():
        ax.axis("off")
    

    for i, (name, rec) in enumerate(reconstruction.items()):
            #axes.ravel()[i + 1 ].imshow(rec.crop((468, 212, 768, 512))) # cropped for easy comparison
        axes.ravel()[i ].imshow(rec)
        axes.ravel()[i].title.set_text(name)

        #plt.show()
    plt.savefig(svpt)
    plt.close()


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics( org, rec, max_val: int = 255):
    metrics =  {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics

def load_models(dict_model_list,  models_path, device, image_models):

    res = {}
    for i, name in enumerate(list(dict_model_list.keys())):#dict_model_listload

        if int(name.split("-")[0][1:]) >= 10:#"q10" in name or "a10" in name:
            nm = name[4:].split(".")[0] # + "-sos"#[3:] # bmshj2018-base/sos
        else: 
            nm = name[3:].split(".")[0] 
        #nm_sos = nm + "-sos"
        #nm_base = nm + "-base"
        print("vedo che tipo di nome è venuto  ",nm,"<-------- ")
        checkpoint =  dict_model_list[name] 
        if "sos" in nm:

            
            architecture =  image_models[nm]
            pt = os.path.join(models_path, checkpoint)
            checkpoint = torch.load(pt, map_location=device)
            N = 192
            M = 320

            factorized_configuration =checkpoint["factorized_configuration"]
            factorized_configuration["trainable"] = True
            gaussian_configuration =  checkpoint["gaussian_configuration"]
            gaussian_configuration["trainable"] = True
            model =architecture(N, M, factorized_configuration = factorized_configuration, gaussian_configuration = gaussian_configuration)
            #print("performing the state dict on ", pt, "     ",checkpoint["state_dict"][ "entropy_bottleneck._offset"].shape)
            model = model.to(device)
            #if checkpoint["state_dict"][ "entropy_bottleneck._offset"].shape !=  torch.Size([0]):               
            model.update( device = device)
            model.load_state_dict(checkpoint["state_dict"])  
            #print("**********************rrr****************************************************************************************") 
            #print("**************************************************************************************************************")  
            model.entropy_bottleneck.sos.update_state(device = device )
            model.gaussian_conditional.sos.update_state(device = device)
            #print(name,"weightsss----> ",model.gaussian_conditional.sos.w)
            


            model.update( device = device)


            
        else:
            #print("il nome è: ", name) #sss
            #model = aux_net_models["cnn_base"]
            qual = int(name.split("-")[0][1:])
            if qual <=-1:
                #print("carico il modello inverso: ",nm,": ",dict_model_list[name])
                
                pt = os.path.join("/scratch/pretrained_models/stf2022",dict_model_list[name] + ".pth.tar")

                
                
                state_dict = load_pretrained(torch.load(pt, map_location=device)['state_dict'])
                model = from_state_dict(aux_net_models["stf"], state_dict) #.eval()

                model.update()
                model.to(device) 


                
                #torch.save({"state_dict": model.state_dict()},"/scratch/inference/baseline_models/zou2022/q1_1905.pth.tar")
            #else:
                #print("pass")

        res[name] = { "model": model}
        #print("HO APPENA FINITO DI CARICARE I MODELLI")
    return res


def modify_dictionary(check):
    res = {}
    ks = list(check.keys())
    for key in ks: 
        res[key[7:]] = check[key]
    return res


def collect_images(rootpath: str):
    image_files = []

    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(rootpath).rglob(f"*{ext}"))
    return sorted(image_files)


def read_image(filepath, clic =False):
    #assert filepath.is_file()
    img = Image.open(filepath)
    
    if clic:
        i =  img.size
        i = i[0]//2, i[1]//2
        img = img.resize(i)
    img = img.convert("RGB")
    return transforms.ToTensor()(img)





transl_sos = {  


            "q17":"17",
            "q21":"21",
            "q22":"22",
            "q23":"23",
            "q26":"26",
            "q31":"31",
            "q32":"32",
            "a10":"10",
            "a30":"30",
            "a60":"60"
              }

transl= {  
            "q6": "18",
              "q5": "22",
              "q4": "27",
              "q3":"32",
              "q2":"37",
              "q1":"42",
              "q7": "15"
              }







@torch.no_grad()
def inference(model, filelist, device, sos,model_name, entropy_estimation = False):
    # tolgo il dataloader al momento
    psnr = AverageMeter()
    ms_ssim = AverageMeter()
    bpps = AverageMeter()
    quality_level =model_name.split("-")[0]
    print("inizio inferenza!")
    i = 0
    for d in filelist:
        name = "image_" + str(i)
        i +=1
        x = read_image(d).to(device)
        x = x.unsqueeze(0) 
        h, w = x.size(2), x.size(3)
        pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
        x_padded = F.pad(x, pad, mode="constant", value=0)
        imgg = transforms.ToPILImage()(x_padded.squeeze())
        
        #d = d.to("cpu")
        #x_padded = d 
        #unpad = 0
        #imgg = transforms.ToPILImage()(d)
        #print("lo shape at encoded is: ",d.shape)
        #data =  model.compress(x_padded)
        if entropy_estimation is False:
            #print("entro qua!!!!")
            data =  model.compress(x_padded)
            if sos: 
                out_dec = model.decompress(data)
            else:
                out_dec = model.decompress(data["strings"], data["shape"])
        else:
            if sos: 
                out_dec = model(x_padded, training = False)
            else: 
                out_dec = model(x_padded)
        if entropy_estimation is False:
            out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
            #print("lo shape decoded è-------------------------------> ",out_dec["x_hat"].shape," ",x.shape)
            out_dec["x_hat"].clamp_(0.,1.)
            metrics = compute_metrics(x, out_dec["x_hat"], 255)
            size = out_dec['x_hat'].size()
            num_pixels = size[0] * size[2] * size[3]
            if sos:
                bpp ,bpp_1, bpp_2= bpp_calculation(out_dec, data["strings"])
            else:
                bpp = sum(len(s[0]) for s in data["strings"]) * 8.0 / num_pixels
            
            #metrics = compute_metrics(x_padded, out_dec["x_hat"], 255)

        else:
            out_dec["x_hat"].clamp_(0.,1.)
            out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
            size = out_dec['x_hat'].size()
            num_pixels = size[0] * size[2] * size[3]
            bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_dec["likelihoods"].values())
            metrics = compute_metrics(x, out_dec["x_hat"], 255)
        
        
        if i <= -1:
            if sos is False:
                image = transforms.ToPILImage()(out_dec['x_hat'].squeeze())
                nome_salv = "/scratch/inference/results/images/devil2022/BASE_2706" + name + model_name +  str(bpp) + "_____" + str(metrics["psnr"]) +  ".png"
                image.save(nome_salv)

                nome_salv2 = "/scratch/inference/results/images/devil2022/" + name + "original" +  ".png"
                imgg.save(nome_salv2)
            else:
                folder_path = "/scratch/inference/images/devil2022/6anchors/" + transl_sos[quality_level] 
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    print(f"Cartella '{folder_path}' creata.")
                #else:
                #    print(f"La cartella '{folder_path}' esiste già.")
                image = transforms.ToPILImage()(out_dec['x_hat'].squeeze())
                nome_salv = os.path.join(folder_path, name + ".png")#"/scratch/inference/images/devil2022/3anchors" + name +    ".png"
                image.save(nome_salv)

        

        psnr.update(metrics["psnr"])
        if i%8==1:
            print(name,": ",metrics["psnr"]," ",bpp, metrics["ms-ssim"])
        ms_ssim.update(metrics["ms-ssim"])
        #bpps.update(bpp.item())
        bpps.update(bpp)

        modality = "None"
        """
        if sos:
            modality = "prop"
            f=open("/scratch/inference/results/clic/bjonte/devil2022/sos_clic_2606_FINALE_6ANCHORS.txt" , "a+")
            f.write("MODE " + modality + " SEQUENCE " + name +  " QP " +  transl_sos[quality_level] + " BITS " +  str(bpp) + " YPSNR " +  str(metrics["psnr"])  + " YMSSIM " +  str(metrics["ms-ssim"]) + "\n")

        else:
            modality = "ref"
            f=open("/scratch/inference/results/tecnik/bjonte/devil2022/devil2022_tecnik_1906_FINALE_BASELINE.txt" , "a+")
            f.write("MODE " + modality + " SEQUENCE " + name +  " QP " +  transl[quality_level] + " BITS " +  str(bpp) + " YPSNR " +  str(metrics["psnr"]) +  " YMSSIM " +  str(metrics["ms-ssim"]) + "\n")
            
        f.close()  
        """
        
    #f=open("/scratch/inference/results/kodak/bjonte/devil2022/FINEGRAINED_0707_kodak_TOTAL.txt" , "a+")
    #f.write("MODE prop" +" SEQUENCE 0 " +  " QP " +  transl_sos[quality_level] + " BITS " +  str(bpps.avg) + " YPSNR " +  str(psnr.avg) +  " YMSSIM " +  str(ms_ssim.avg) + "\n")
    #f.close()      
    print("fine inddferenza",psnr.avg, ms_ssim.avg, bpps.avg)
    return psnr.avg, ms_ssim.avg, bpps.avg


def bpp_calculation(out_net, out_enc):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]

        bpp_1 = (len(out_enc[0]) * 8.0 ) / num_pixels
        #print("la lunghezza è: ",len(out_enc[1]))
        bpp_2 =  sum( (len(out_enc[1][i]) * 8.0 ) / num_pixels for i in range(len(out_enc[1])))
        return bpp_1 + bpp_2, bpp_1, bpp_2



def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

@torch.no_grad()
def eval_models(res, dataloader, device, entropy_estimation):
  
    metrics = {}
    models_name = list(res.keys())
    for i, name in enumerate(models_name): #name = q1-bmshj2018-base/fact
        #print("----")
        print("name: ",name)
        qual = int(name.split("-")[0][1:])
        model = res[name]["model"]
        if "sos" in name: 
            sos = True
        else:
            sos = False

        #if qual <= 6 and "base" not in name:
        #if "base" not in name or qual > 6:
        if  sos is True:
               #asos is False
            if qual > 50:#in (1,2,3,5,7,8):#in (2,5,6,8,7,1,3,4,9):
                psnr, mssim, bpp =  inference(model,dataloader,device, sos, name, entropy_estimation= entropy_estimation)

                metrics[name] = {"bpp": bpp,
                            "mssim": mssim,
                            "psnr": psnr
                                } 
            #else:          
            #    print("non considero questo modello ",sos,"   ",qual)
        else:
            if qual <  -1: 
                psnr, mssim, bpp = inference(model,dataloader,device, sos, name,  entropy_estimation= entropy_estimation)

                metrics[name] = {"bpp": bpp,
                            "mssim": mssim,
                            "psnr": psnr
                                }
    return metrics   

def from_state_dict(cls, state_dict):
    net = cls()
    net.load_state_dict(state_dict)
    return net


def load_pretrained(state_dict):
    """Convert sccctaddte_dict keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    if None in state_dict:
        state_dict.pop(None)
    return state_dict


def load_only_baselines(  model_name, device):
    res = {}
    quality = [1,2,3,4,5,6]
    for qual in quality:

        model = image_models[model_name]

        model = model()

        pattern = "/scratch/pretrained_models/stf2022/q" + str(qual) + "-zou22.pth.tar"
            

        state_dict = load_pretrained(torch.load(pattern, map_location=device)['state_dict'])
        model = from_state_dict(image_models[model_name], state_dict)

        model.update()

        #checkpoint = torch.load(pattern, map_location=device)
        #model.load_state_dict(checkpoint)
        #model.update(force = True)
        nome_completo = "q" + str(qual) + "-zou22"

        res[nome_completo] = { "model": model}
        print("i modelli sono stati tutti caricati: ",nome_completo)
    return res

def extract_specific_model_performance(metrics, name):

    nms = list(metrics.keys())




    psnr = []
    mssim = []
    bpp = []
    for names in nms:
        if name in names:
            psnr.append(metrics[names]["psnr"])
            mssim.append(metrics[names]["mssim"])
            bpp.append(metrics[names]["bpp"])
    
    return sorted(psnr), sorted(mssim), sorted(bpp)







     


def main(argv):
    set_seed()
    args = parse_args(argv)
    model_name = args.model  # nome del modello che voglio studiare (ad esempio cheng2020)
    models_path = join(args.model_path,model_name) # percorso completo per arrivare ai modelli salvati (/scratch/inference/pretrained_models/chegn2020) qua ho salvato i modelli 
 

    models_checkpoint = listdir(models_path) # checkpoints dei modelli  q1-bmshj2018-sos.pth.tar, q2-....
    print(models_checkpoint)
    device = "cpu"
    entropy_estimation = args.entropy_estimation
    
    images_path = args.image_path # path del test set 
    #savepath = args.result_path # path dove salvare i risultati 

    image_list = [os.path.join(images_path,f) for f in listdir(images_path)]
    
    test_dataset = TestKodakDataset(data_dir= images_path ) #test set 
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=4) # data loader 

    dict_model_list =  {} #  inizializzo i modelli 


    # RICORDARSI DI METTERE I BASE QUANDO ARRIVERA' il MOMENTO!!!!!!
    for i, check in enumerate(models_checkpoint):  # per ogni cjeckpoint, salvo il modello nostro con la chiave q1-bmshj2018-sos ed il modello base con la chiave q1-bmshj2018-base (il modello base non ha checkpoint perchè lo prendo online)
        if True: #"q1" in check:
            name_sos = check.split("-")[0] + "-" + check.split("-")[1]  # q1-bmshj2018 
            print("name_sos è il seguente: ",name_sos)
            name_base = name_sos + "-base"  # q1-bmshj2018-base
            print(i,": ",name_base,"  ",name_sos)
            dict_model_list[name_sos + "-sos"] = check
            dict_model_list[name_base] = name_sos 
            




    res = load_models(dict_model_list,  models_path, device, image_models) # carico i modelli res è un dict che ha questa struttura res[q1-bmshj2018-sos] = {"model": model}
    #res = load_only_baselines(  "zou22-base", device)

    # cambiato con test_dataloader
    metrics = eval_models(res,image_list , device,entropy_estimation) #faccio valutazione dei modelli 

    #print("olaaaaaaaaaaaaaaaaaa")
    #compute_time(image_list)


    list_names = list(metrics.keys()) 
    
    #list_names =["xie21-base"]

    #plot_rate_distorsion(metrics,list_names,savepath) # plot del rate distorsion 
    print("ALL DONE!!!!!")

    """
    for filepath in list_images:
        reconstruct_image_with_nn(res, filepath,device,  imagesavepath )
    """


    
if __name__ == "__main__":

    wandb.init(project="prova", entity="albertopresta")   
    main(sys.argv[1:])
