
import torch 
import wandb









def plot_sos(model, device,n = 1000, dim = 0,aq = False):


    if aq is False:
        """
        x_min = float((min(model.entropy_bottleneck.sos.b) + min(model.entropy_bottleneck.sos.b)*0.5).detach().cpu().numpy())
        x_max = float((max(model.entropy_bottleneck.sos.b)+ max(model.entropy_bottleneck.sos.b)*0.5).detach().cpu().numpy())
        step = (x_max-x_min)/n
        x_values = torch.arange(x_min, x_max, step)
        x_values = x_values.repeat(model.entropy_bottleneck.M,1,1)
            
        print("entro qua spero!")
        y_values= model.entropy_bottleneck.sos(x_values.to(device))[0,0,:]
        data = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
        table = wandb.Table(data=data, columns = ["x", "sos"])
        wandb.log({"FactorizedSoS/SoS at dimension " + str(dim): wandb.plot.line(table, "x", "sos", title='FactorizedSoS/SoS  with beta = {}'.format(model.entropy_bottleneck.sos.beta))})
        y_values= model.entropy_bottleneck.sos(x_values.to(device), -1)[0,0,:]
        data_inf = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
        table_inf = wandb.Table(data=data_inf, columns = ["x", "sos"])
        wandb.log({"FactorizedSoS/SoS  inf at dimension " + str(dim): wandb.plot.line(table_inf, "x", "sos", title='FactorizedSoS/SoS  with beta = {}'.format(-1))})  

        """
        x_min = float((min(model.gaussian_conditional.sos.b) + min(model.gaussian_conditional.sos.b)*0.5).detach().cpu().numpy())
        x_max = float((max(model.gaussian_conditional.sos.b)+ max(model.gaussian_conditional.sos.b)*0.5).detach().cpu().numpy())
        step = (x_max-x_min)/n
        x_values = torch.arange(x_min, x_max, step)
        x_values = x_values.repeat(model.gaussian_conditional.M,1,1)
            
        y_values=model.gaussian_conditional.sos(x_values.to(device))[0,0,:]
        data = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
        table = wandb.Table(data=data, columns = ["x", "sos"])
        wandb.log({"GaussianSoS/Gaussian SoS at dimension " + str(dim): wandb.plot.line(table, "x", "sos", title='GaussianSoS/Gaussian SoS  with beta = {}'.format(model.gaussian_conditional.sos.beta))})
        y_values= model.gaussian_conditional.sos(x_values.to(device), -1)[0,0,:]
        data_inf = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
        table_inf = wandb.Table(data=data_inf, columns = ["x", "sos"])
        wandb.log({"GaussianSoS/Gaussian SoS  inf at dimension " + str(dim): wandb.plot.line(table_inf, "x", "sos", title='GaussianSoS/Gaussian SoS  with beta = {}'.format(-1))})  


