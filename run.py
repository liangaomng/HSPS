#!/bin/sh
''''exec python -u -- "$0" "$@" # '''
# vi: syntax=python

"""
run.py

Main script for training or evaluating a PDE-VAE model specified by the input file (JSON format).

Usage:
python run.py input_file.json > out
"""
from scipy.stats import multivariate_normal, norm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys
from shutil import copy2
import json
from types import SimpleNamespace
import warnings

import numpy as np
from matplotlib.ticker import ScalarFormatter

from matplotlib.ticker import FuncFormatter

import torch
import torch.nn as nn
import torch.nn.functional as F
# 设置全局默认数据类型为float32
torch.set_default_dtype(torch.float32)
tsne = TSNE(n_components=2, random_state=0)
from torch.utils.tensorboard import SummaryWriter

def setup(in_file):
    # Load configuration from json
    with open(in_file) as f:
        s = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    # Some defaults
    if not hasattr(s, 'train'):
        raise NameError("'train' must be set to True for training or False for evaluation.")
    elif s.train == False and not hasattr(s, 'MODELLOAD'):
        raise NameError("'MODELLOAD' file name required for evaluation.")

    if not hasattr(s, 'restart'):
        s.restart = not s.train
        warnings.warn("Automatically setting 'restart' to " + str(s.restart))
    if s.restart and not hasattr(s, 'MODELLOAD'):
        raise NameError("'MODELLOAD' file name required for restart.")

    if not hasattr(s, 'freeze_encoder'):
        s.freeze_encoder = False
    elif s.freeze_encoder and not s.restart:
        raise ValueError("Freeezing encoder weights requires 'restart' set to True with encoder weights loaded from file.")

    if not hasattr(s, 'data_parallel'):
        s.data_parallel = False
    if not hasattr(s, 'debug'):
        s.debug = False
    if not hasattr(s, 'discount_rate'):
        s.discount_rate = 0.
    if not hasattr(s, 'rate_decay'):
        s.rate_decay = 0.
    if not hasattr(s, 'param_dropout_prob'):
        s.param_dropout_prob = 0.
    if not hasattr(s, 'prop_noise'):
        s.prop_noise = 0.
    if not hasattr(s,"prior_mu"):
        s.prior_mu =0
    if not hasattr(s,"prior_std"):
        s.prior_std =1

    if not hasattr(s, 'boundary_cond'):
        raise NameError("Boundary conditions 'boundary_cond' not set. Options include: 'crop', 'periodic', 'dirichlet0'")
    elif s.boundary_cond == 'crop' and (not hasattr(s, 'input_size') or not hasattr(s, 'training_size')):
        raise NameError("'input_size' or 'training_size' not set for crop boundary conditions.")

    # Create output folder
    if not os.path.exists(s.OUTFOLDER):
        print("Creating output folder: " + s.OUTFOLDER)
        os.makedirs(s.OUTFOLDER)
    elif s.train and os.listdir(s.OUTFOLDER):
        raise FileExistsError("Output folder " + s.OUTFOLDER + " is not empty.")

    # Make a copy of the configuration file in the output folder
    copy2(in_file, s.OUTFOLDER)

    # Print configuration
    print(s)

    # Import class for dataset type
    dataset = __import__(s.dataset_type, globals(), locals(), ['PDEDataset'])
    s.PDEDataset = dataset.PDEDataset

    # Import selected model from models as PDEModel
    models = __import__('models.' + s.model, globals(), locals(), ['PDEAutoEncoder'])
    PDEModel = models.PDEAutoEncoder

    # Initialize model
    model = PDEModel(param_size=s.param_size, data_channels=s.data_channels, data_dimension=s.data_dimension,
                    hidden_channels=s.hidden_channels, linear_kernel_size=s.linear_kernel_size, 
                    nonlin_kernel_size=s.nonlin_kernel_size, prop_layers=s.prop_layers, prop_noise=s.prop_noise,
                    boundary_cond=s.boundary_cond, param_dropout_prob=s.param_dropout_prob, debug=s.debug)
    
    
    # Set CUDA device
    s.use_cuda = torch.cuda.is_available()
    if s.use_cuda:
        print("Using cuda device(s): " + str(s.cuda_device))
        torch.cuda.set_device(s.cuda_device)
        model.cuda()
    else:
        warnings.warn("Warning: Using CPU only. This is untested.")
        print("hi",flush=True)
    model =model.float()

    # print("\nModel parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print("\t{:<40}{}".format(name + ":", param.shape))

    return model, s


def _periodic_pad_1d(x, dim, pad):
    back_padding = x.narrow(dim, 0, pad)
    return torch.cat((x, back_padding), dim=dim)


def _random_crop_1d(sample, depth, crop_size):
    sample_size = sample[0].shape
    crop_t = [np.random.randint(sample_size[-2]-depth[0]+1), np.random.randint(sample_size[-2]-depth[1]+1)]
    crop_x = [np.random.randint(sample_size[-1]), np.random.randint(sample_size[-1])]
   
    if crop_size[0] > 1: 
        sample[0] = _periodic_pad_1d(sample[0], -1, crop_size[0]-1)
    if crop_size[1] > 1:
        sample[1] = _periodic_pad_1d(sample[1], -1, crop_size[1]-1)

    if len(sample_size) == 3:
        sample[0] = sample[0][:, crop_t[0]:(crop_t[0]+depth[0]), crop_x[0]:(crop_x[0]+crop_size[0])]
        sample[1] = sample[1][:, crop_t[1]:(crop_t[1]+depth[1]), crop_x[1]:(crop_x[1]+crop_size[1])]
    elif len(sample_size) == 2:
        sample[0] = sample[0][crop_t[0]:(crop_t[0]+depth[0]), crop_x[0]:(crop_x[0]+crop_size[0])]
        sample[1] = sample[1][crop_t[1]:(crop_t[1]+depth[1]), crop_x[1]:(crop_x[1]+crop_size[1])]
    else:
        raise ValueError('Sample is the wrong shape.')
        
    return sample


def _random_crop_2d(sample, depth, crop_size):
    sample_size = sample[0].shape
    crop_t = [np.random.randint(sample_size[-3]-depth[0]+1), np.random.randint(sample_size[-3]-depth[1]+1)]
    crop_x = [np.random.randint(sample_size[-2]), np.random.randint(sample_size[-2])]
    crop_y = [np.random.randint(sample_size[-1]), np.random.randint(sample_size[-1])]
    
    if crop_size[0] > 1:
        sample[0] = _periodic_pad_1d(_periodic_pad_1d(sample[0], -1, crop_size[0]-1), -2, crop_size[0]-1)
    if crop_size[1] > 1:
        sample[1] = _periodic_pad_1d(_periodic_pad_1d(sample[1], -1, crop_size[1]-1), -2, crop_size[1]-1)

    if len(sample_size) == 4:
        sample[0] = sample[0][:, crop_t[0]:(crop_t[0]+depth[0]), crop_x[0]:(crop_x[0]+crop_size[0]), crop_y[0]:(crop_y[0]+crop_size[0])]
        sample[1] = sample[1][:, crop_t[1]:(crop_t[1]+depth[1]), crop_x[1]:(crop_x[1]+crop_size[1]), crop_y[1]:(crop_y[1]+crop_size[1])]
    elif len(sample_size) == 3:
        sample[0] = sample[0][crop_t[0]:(crop_t[0]+depth[0]), crop_x[0]:(crop_x[0]+crop_size[0]), crop_y[0]:(crop_y[0]+crop_size[0])]
        sample[1] = sample[1][crop_t[1]:(crop_t[1]+depth[1]), crop_x[1]:(crop_x[1]+crop_size[1]), crop_y[1]:(crop_y[1]+crop_size[1])]
    else:
        raise ValueError('Sample is the wrong shape.')

    return sample

def prepare_and_display(ax, img_data, title, index, channel=1,diff=False, vmin=-1, vmax=1, cmap='bwr'):
    
    length = img_data.shape[-1]
    width = img_data.shape[-2]

  
    img = img_data.cpu().detach().numpy().reshape(-1,width, length)
    if channel in [1, 2]:
        # 假设img已经是两个通道的数据，直接选取对应通道
        img_channel = img[channel - 1]  # 通道索引调整为从0开始
        cb = ax.imshow(img_channel, vmin=vmin, vmax=vmax, cmap=cmap)

        if diff:
            mse = np.mean(img_channel**2)  # 使用numpy计算MSE
            ax.set_title(f"{title}_mse{mse:.2e}",fontsize=8)
        else:
            ax.set_title(title,fontsize=8)

        # 仅在每行的最后一个图像添加颜色条
        num_cols = 3  # 你的布局中每行的图像数
        if (index + 1) % num_cols == 0:
            plt.colorbar(cb, ax=ax, fraction=0.03, pad=0.04)

               
def plot_var_params(mu_data,logvar):
    # 假设这些是你的数据
    # mu_data: [batch,5]
    # 平均值数据
    latent_numbers = mu_data.shape[-1]
    mu_mean  = torch.mean(mu_data,dim=0).cpu().detach().numpy() 
    mu_variances = mu_data.var(dim=0, unbiased=False) .cpu().detach().numpy() # `unbiased=False` 用于计算样本方差
    logvar_mean = logvar.mean(dim=0).cpu().detach().numpy()
    labels = ['1', '2', '3', '4', '5']  # 潜在参数标签

    # 设置柱状图的位置和宽度
    x = np.arange(latent_numbers)  # label位置
    width = 0.35  # 柱状图的宽度

    fig, ax = plt.subplots(1,2,figsize=(12,6))

    # 绘制方差数据的柱状图
    rects1 = ax[0].bar(x - width/2, mu_variances, width, label='Var of mean', color='tab:blue')

    # 使用相同的x轴，但不同的y轴绘制平均值数据的柱状图
    ax2 = ax[0].twinx()
    rects2 = ax2.bar(x + width/2, np.exp(logvar_mean), width, label='Mean of var', color='tab:red')

    # 添加标签、标题和自定义x轴刻度标签
    ax[0].set_xlabel('Latent parameters')
    ax[0].set_ylabel('Var(µ)', color='tab:blue')
    ax2.set_ylabel('Mean(σ^2)', color='tab:red')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels[:latent_numbers])
    ax[0].tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 添加图例

    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 0.85))


    # 使用列表推导式和字符串的join方法来创建格式化的字符串
    mu_str = ", ".join([f"{mu:.1e}" for mu in mu_mean])
    logvar_str = ", ".join([f"{np.exp(logvar):.1e}" for logvar in logvar_mean])

    # 将这些字符串组合并设置为标题
    title_str = f'Batch Gaussian_µ: [{mu_str}] \n var: [{logvar_str}]'
    if mu_data.shape[-1]==2:
        # 二维高斯分布
        # 创建网格坐标
        x, y = np.mgrid[-3:3:.01, -3:3:.01]
        pos = np.dstack((x, y))
        rv = multivariate_normal(mu_mean,  np.sqrt(np.exp(logvar_mean)))
        z = rv.pdf(pos)
        ax[1].contourf(x, y, z, cmap='Blues')
        ax[1].set_title(title_str)
        ax[1].set_xlabel('X-axis')
        ax[1].set_ylabel('Probability density')
    elif mu_data.shape[-1]==1:
        # 计算一维高斯分布
        # 创建一维坐标
        x_1d = np.linspace(-5, 5, 1000)
        sigma_std= np.sqrt(np.exp(logvar_mean))
        pdf = norm(mu_mean,  np.sqrt(np.exp(logvar_mean))).pdf(x_1d)

        # 绘制一维高斯分布 先验
        sigma_P= s.prior_std
        mu_p = s.prior_mu
        pri_pdf = norm( mu_p,  sigma_P).pdf(x_1d)
        ax[1].plot(x_1d, pdf,c="k", linestyle='--',alpha=0.8, lw=4,label=f"NN inference,mu_{mu_mean[0]:.2e}_sigma_{sigma_std[0]:.2e}")
        ax[1].plot(x_1d, pri_pdf, 'blue',alpha=0.8, lw=3,label=f"Prior,mu_{mu_p}_sigma_{sigma_P}")
        ax[1].set_title(title_str)
        ax[1].set_xlabel('X-axis')  
        ax[1].set_ylabel('Probability density')
    elif mu_data.shape[-1]>2:
        # Initialize a t-SNE object.
        # Reduce the dimensionality of the data.
        data_reduced = tsne.fit_transform(mu_data.cpu().detach().numpy() )
        ax[1].scatter(data_reduced[:, 0], data_reduced[:, 1], c=data_reduced[:, 0], cmap='rainbow')
        ax[1].set_title(title_str)
        ax[1].set_xlabel('Dimension1')
        ax[1].set_ylabel('Dimension2')
    ax[1].legend(loc="upper left")

   



def train(model, s):
    ### Train model on training set
    print("\nTraining...")

    if s.restart: # load model to restart training
        print("Loading model from: " + s.MODELLOAD)
        strict_load = not s.freeze_encoder
        if s.use_cuda:
            state_dict = torch.load(s.MODELLOAD, map_location=torch.device('cuda', torch.cuda.current_device()))
        else:
            state_dict = torch.load(s.MODELLOAD)
        model.load_state_dict(state_dict, strict=strict_load)

        if s.freeze_encoder: # freeze encoder weights
            print("Freezing weights:")
            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
                print("\t{:<40}{}".format("encoder." + name + ":", param.size()))
            for name, param in model.encoder_to_param.named_parameters():
                param.requires_grad = False
                print("\t{:<40}{}".format("encoder_to_param." + name + ":", param.size()))
            for name, param in model.encoder_to_logvar.named_parameters():
                param.requires_grad = False
                print("\t{:<40}{}".format("encoder_to_logvar." + name + ":", param.size()))

    if s.data_parallel:
        model = nn.DataParallel(model, device_ids=s.cuda_device)

    if s.boundary_cond == 'crop':
        if s.data_dimension == 1:
            transform = lambda x: _random_crop_1d(x, (s.input_depth, s.training_depth+1), (s.input_size, s.training_size)) 
        elif s.data_dimension == 2:
            transform = lambda x: _random_crop_2d(x, (s.input_depth, s.training_depth+1), (s.input_size, s.training_size)) 
        
        pad = int((2+s.prop_layers)*(s.nonlin_kernel_size-1)/2) #for cropping targets

    elif s.boundary_cond == 'periodic' or s.boundary_cond == 'dirichlet0':
        transform = None

    else:
        raise ValueError("Invalid boundary condition.")

    train_loader = torch.utils.data.DataLoader(
        s.PDEDataset(data_file=s.DATAFILE, transform=transform),
        batch_size=s.batch_size, shuffle=True, num_workers=s.num_workers, pin_memory=True,
        worker_init_fn=lambda _: np.random.seed())

    optimizer = torch.optim.Adam(model.parameters(), lr=s.learning_rate, eps=s.eps)

    model.train()

    writer = SummaryWriter(log_dir=os.path.join(s.OUTFOLDER, 'data'))

    # Initialize training variables
    loss_list = []
    recon_loss_list = []
    mse_list = []
    acc_loss = 0
    acc_recon_loss = 0
    acc_latent_loss = 0
    acc_mse = 0
    best_mse = None
    step = 0
    current_discount_rate = s.discount_rate

    ### Training loop
    for epoch in range(1, s.max_epochs+1):
        print('\nEpoch: ' + str(epoch))
        
        # Introduce a discount rate to favor predicting better in the near future
        current_discount_rate = s.discount_rate * np.exp(-s.rate_decay * (epoch-1)) # discount rate decay every epoch
        print('discount rate = ' + str(current_discount_rate))
        if current_discount_rate > 0:
            w = torch.tensor(np.exp(-current_discount_rate * np.arange(s.training_depth)).reshape(
                    [s.training_depth] + s.data_dimension * [1]), dtype=torch.float32, device='cuda' if s.use_cuda else 'cpu')
            w = w * s.training_depth/w.sum(dim=0, keepdim=True)
        else:
            w = None


        # Load batch and train
        for data, target, data_params in train_loader:
            step += 1
            

            if s.use_cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                data = data.float()
                target= target.float()
            

            data = data[:,:,:s.input_depth]
            
            if s.boundary_cond == 'crop':
                target0 = target[:,:,:s.training_depth]
                if s.data_dimension == 1:
                    target = target[:,:,1:s.training_depth+1, pad:-pad]
                elif s.data_dimension == 2:
                    target = target[:,:,1:s.training_depth+1, pad:-pad, pad:-pad]
            
            elif s.boundary_cond == 'periodic' or s.boundary_cond == 'dirichlet0':
                target0 = target[:,:,1]


                target = target[:,:,1:s.training_depth+1]

            else:
                raise ValueError("Invalid boundary condition.")

            # Run model

           
            output, params, logvar = model(data, target0, depth=s.training_depth) # out : 时间=199
            print(output.shape)

            if s.data_channels==1:
                fig,ax = plt.subplots(3,3,figsize=(12,6))
                # 将ax数组扁平化，以便可以使用索引访问每个子图
                ax_flat = ax.flatten()
            elif s.data_channels==2:

                fig,ax = plt.subplots(6,3,figsize=(8,8))
                
                ax_flat = ax.flatten()


            # Indices to select specific slices for Pred and True sections
            pred_indices = [0, 15, 35]
            true_indices = [0, 15, 35]

            for i, (pred_idx, true_idx) in enumerate(zip(pred_indices, true_indices)):
                # 旋转量的预测、真实和差异图像的索引
                vorticity_pred_index = i
                vorticity_true_index = 6 + i  # 旋转量的真实图像放在6, 7, 8
                vorticity_diff_index = 12+i

                # 压力的预测、真实和差异图像的索引
                pressure_pred_index = 3 + i  # 压力的预测图像放在3, 4, 5
                pressure_true_index = 9 + i  # 压力的真实图像放在9, 10, 11
                pressure_diff_index = 15+i

                # 针对旋转量（Vorticity）
                if s.data_channels >= 1:
                    # 显示旋转量的预测图
                    prepare_and_display(ax_flat[vorticity_pred_index], output[0, :, pred_idx, :, :], title=f"{epoch}_Pred_V_t{pred_idx}", index=vorticity_pred_index, channel=1)
                    # 显示旋转量的真实图
                    prepare_and_display(ax_flat[vorticity_true_index], target[0, :, true_idx, :, :], title=f"True_V_t{true_idx}", index=vorticity_true_index, channel=1)
                    # 显示旋转量的差异图
                    diff_img_v = output[0, :, pred_idx, :, :] - target[0, :, true_idx, :, :]
                    prepare_and_display(ax_flat[vorticity_diff_index], diff_img_v, title=f"Diff_V_t{true_idx}", index=vorticity_diff_index, diff=True, channel=1)

                # 针对压力（Pressure）
                if s.data_channels == 2:
                    # 显示压力的预测图
                    prepare_and_display(ax_flat[pressure_pred_index], output[0, :, pred_idx, :, :], title=f"{epoch}_Pred_P_t{pred_idx}", index=pressure_pred_index, channel=2)
                    # 显示压力的真实图
                    prepare_and_display(ax_flat[pressure_true_index], target[0, :, true_idx, :, :], title=f"True_P_t{true_idx}", index=pressure_true_index, channel=2)
                    # 显示压力的差异图
                    diff_img_p = output[0, :, pred_idx, :, :] - target[0, :, true_idx, :, :]
                    prepare_and_display(ax_flat[pressure_diff_index], diff_img_p, title=f"Diff_P_t{true_idx}", index=pressure_diff_index, diff=True, channel=2)

            plt.tight_layout()
            fig.subplots_adjust(wspace=0.4)

            plt.savefig(f"{s.OUTFOLDER}""/"+f"{epoch}_perform.png",dpi=200)
            plt.close()

            plot_var_params(mu_data=params,logvar=logvar)
            plt.tight_layout()
            plt.savefig(f"{s.OUTFOLDER}""/"+f"{epoch}_var_params.png",dpi=200)
            plt.close()
            # Reset gradients
            optimizer.zero_grad()

            # Calculate loss
            if s.data_parallel:
                output = output.cpu()

            recon_loss =  F.mse_loss(output * w, target * w) if w is not None else F.mse_loss(output, target)

            if s.param_size > 0:
                 # 先计算后验方差sigma^2 (logvar.exp())和先验方差的比例，以及后验均值与先验均值的差的平方
                variance_term = logvar.exp() / (s.prior_std ** 2)
                mean_term = ((params - s.prior_mu) ** 2) / (s.prior_std ** 2)
                s.prior_std_torch = torch.tensor(s.prior_std, device=params.device)
                
                # 然后计算每个维度的KL散度，最后求平均得到整体的KL散度损失
                latent_loss = s.beta * 0.5 * torch.mean(torch.sum(variance_term + mean_term - 1 - logvar + 2 * torch.log(s.prior_std_torch), dim=-1))
                
            else:
                latent_loss = 0
            loss = recon_loss + latent_loss
            print("latent",latent_loss.item())

            mse = F.mse_loss(output.detach(), target.detach()).item() if w is not None else recon_loss.item()
            

            loss_list.append(loss.item())
            recon_loss_list.append(recon_loss.item())
            mse_list.append(mse)

            acc_loss += loss.item()
            acc_recon_loss += recon_loss.item()
            acc_latent_loss += latent_loss.item()
            acc_mse += mse

            # Calculate gradients
            loss.backward()
            print(f"{step}_recon_loss",recon_loss.item(),flush=True)

            # Clip gradients
            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1e0)

            # Update gradients
            optimizer.step()

            # Output every 100 steps
            if step % 100 == 0:
                # Check every 500 steps and save checkpoint if new model is at least 2% better than previous best
                if (step > 1 and step % 100 == 0) and ((best_mse is None) or (acc_mse/100 < 0.98*best_mse)):
                    best_mse = acc_mse/100
                    torch.save(model.state_dict(), os.path.join(s.OUTFOLDER, "best.tar"))
                    print('New Best MSE at Step {}: {:.4f}'.format(step, best_mse))

                # Output losses and weights
                if s.param_size > 0:
                    if step > 1:
                        # Write losses to summary
                        writer.add_scalars('losses',    {'loss': acc_loss/100,
                                                         'recon_loss': acc_recon_loss/100,
                                                         'latent_loss': acc_latent_loss/100,
                                                         'mse': acc_mse/100}, step)

                        acc_loss = 0
                        acc_recon_loss = 0
                        acc_latent_loss = 0
                        acc_mse = 0

                    # Write mean model weights to summary
                    weight_dict = {}
                    # for name, param in model.named_parameters():
                    #     if param.requires_grad:
                    #         weight_dict[name] = param.detach().abs().mean().item()
                    # writer.add_scalars('weight_avg', weight_dict, step)

                    print('Train Step: {}\tTotal Loss: {:.4f}\tRecon. Loss: {:.4f}\tRecon./Latent: {:.1f}\tMSE: {:.4f}'
                            .format(step, loss.item(), recon_loss.item(), recon_loss.item()/latent_loss.item(), mse))
                    
                    # Save current set of extracted latent parameters
                    np.savez(os.path.join(s.OUTFOLDER, "training_params.npz"),  data_params=data_params.numpy(), 
                                                                                params=params.detach().cpu().numpy())
                else:
                    print('Train Step: {}\tTotal Loss: {:.4f}\tRecon. Loss: {:.4f}\tMSE: {:.4f}'
                            .format(step, loss.item(), recon_loss.item(), mse))

        # Export checkpoints and loss history after every s.save_epochs epochs
        if s.save_epochs > 0 and epoch % s.save_epochs == 0:
            torch.save(model.state_dict(), os.path.join(s.OUTFOLDER, "epoch{:06d}.tar".format(epoch)))
            np.savez(os.path.join(s.OUTFOLDER, "loss.npz"), loss=np.array(loss_list), 
                                                            recon_loss=np.array(recon_loss_list), 
                                                            mse=np.array(mse_list))

    return model


def evaluate(model, s, params_filename="params.npz", rmse_filename="rmse_with_depth.npy"):
    ### Evaluate model on test set
    print("\nEvaluating...")

    if rmse_filename is not None and os.path.exists(os.path.join(s.OUTFOLDER, rmse_filename)):
        raise FileExistsError(rmse_filename + " already exists.")
    if os.path.exists(os.path.join(s.OUTFOLDER, params_filename)):
        raise FileExistsError(params_filename + " already exists.")

    if not s.train:
        print("Loading model from: " + s.MODELLOAD)
        if s.use_cuda:
            state_dict = torch.load(s.MODELLOAD, map_location=torch.device('cuda', torch.cuda.current_device()))
        else:
            state_dict = torch.load(s.MODELLOAD)
        model.load_state_dict(state_dict)
        
    pad = int((2+s.prop_layers)*(s.nonlin_kernel_size-1)/2) #for cropping targets (if necessary)

    test_loader = torch.utils.data.DataLoader(
        s.PDEDataset(data_file=s.DATAFILE, transform=None),
        batch_size=s.batch_size, num_workers=s.num_workers, pin_memory=True)

    model.eval()
    torch.set_grad_enabled(False)

    ### Evaluation loop
    loss = 0
    if rmse_filename is not None:
        rmse_with_depth = torch.zeros(s.evaluation_depth, device='cuda' if s.use_cuda else 'cpu')
    params_list = []
    logvar_list = []
    data_params_list = []
    step = 0
    for data, target, data_params in test_loader:
        step += 1

        if s.use_cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        if s.boundary_cond == 'crop':
            target0 = target[:,:,:s.evaluation_depth]
            if s.data_dimension == 1:
                target = target[:,:,1:s.evaluation_depth+1, pad:-pad]
            elif s.data_dimension == 2:
                target = target[:,:,1:s.evaluation_depth+1, pad:-pad, pad:-pad]
        
        elif s.boundary_cond == 'periodic' or s.boundary_cond == 'dirichlet0':
            target0 = target[:,:,0]
            target = target[:,:,1:s.evaluation_depth+1]

        else:
            raise ValueError("Invalid boundary condition.")

        # Run model
        if s.debug:
            output, params, logvar, _, weights, raw_params = model(data.contiguous(), target0, depth=s.evaluation_depth)
        else:
            output, params, logvar = model(data.contiguous(), target0, depth=s.evaluation_depth)

        data_params = data_params.numpy()
        data_params_list.append(data_params)

        if s.param_size > 0:
            params = params.detach().cpu().numpy()
            params_list.append(params)
            logvar_list.append(logvar.detach().cpu().numpy())

        assert output.shape[2] == s.evaluation_depth
        loss += F.mse_loss(output, target).item()

        if rmse_filename is not None:
            rmse_with_depth += torch.sqrt(torch.mean((output - target).transpose(2,1).contiguous()
                                        .view(target.size()[0], s.evaluation_depth, -1) ** 2,
                                                 dim=-1)).mean(0)

    rmse_with_depth = rmse_with_depth.cpu().numpy()/step
    print('\nTest Set: Recon. Loss: {:.4f}'.format(loss/step))

    if rmse_filename is not None:
        np.save(os.path.join(s.OUTFOLDER, rmse_filename), rmse_with_depth)

    np.savez(os.path.join(s.OUTFOLDER, params_filename), params=np.concatenate(params_list), 
                                                         logvar=np.concatenate(logvar_list),
                                                         data_params=np.concatenate(data_params_list))


if __name__ == "__main__":
    in_file = sys.argv[1]
    if not os.path.exists(in_file):
        raise FileNotFoundError("Input file " + in_file + " not found.")

    model, s = setup(in_file)
    if s.train:
        print("train")
        model = train(model, s)
    else:
        print("eva")
        evaluate(model, s)
