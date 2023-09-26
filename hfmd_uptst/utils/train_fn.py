from utils.metrics import metric
from utils.tools import visual
import torch
import os
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import torch.nn as nn
from utils.tools import EarlyStopping,adjust_learning_rate
from torch.cuda.amp import autocast, GradScaler

def train_model(
    args,model, train_dataloader, val_dataloader,test_dataloader,pre_dim
    ):
    alpha_1, alpha_2 = args.alpha,1-args.alpha
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    best_val_loss = float('inf')
    print('\n Training Start!')
    print("Model total parameters: {:.2f} M".format(sum(p.numel() for p in model.parameters())/1e+6))
    early_stopping = EarlyStopping(patience=args.patience, verbose=True,save=True)
    scaler = GradScaler(enabled=args.use_mixed_precision)  # Create a GradScaler object for mixed precision training
    
    check_head=args.checkpoint_path+f'/uptst/{args.seed}/{args.seq_len}->{args.pre_len}/a1_{alpha_1}/'
    check_tail='/'+args.loss_fn+'/'+f'bs_{args.bs}/'+f'lr_{args.lr}'
    visual_head=f'./visual/{args.seed}/{args.seq_len}->{args.pre_len}/a1_{alpha_1}/'
    visual_tail=check_tail
    dict_head=f'./test_dict/{args.seed}/{args.seq_len}->{args.pre_len}/a1_{alpha_1}/'
    dict_tail='/'+args.loss_fn+'/'+f'bs_{args.bs}/lr_{args.lr}/'
    
    check_path=check_head  +args.tasks+f'/revin_{args.revin}'+  check_tail
    visual_folder=visual_head  +args.tasks+ f'/revin_{args.revin}'+ visual_tail
    save_folder=dict_head  +args.tasks+ f'/revin_{args.revin}'+ dict_tail
    
        
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(check_path):
        os.makedirs(check_path)
    if not os.path.exists(visual_folder):
        os.makedirs(visual_folder)
    if args.loss_fn=='mse':
        loss_fn=nn.MSELoss()
    elif args.loss_fn=='mae':
        loss_fn=nn.L1Loss()
    for epoch in range(args.epochs):
        total_loss = 0.0
        loss_term_1 = 0.0
        loss_term_2 = 0.0
        model=model.to(args.device)
        model.train()
        for batch_idx, (input, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            # Move data to GPU
            input, target = input.to(args.device), target.to(args.device)
            with autocast(enabled=args.use_mixed_precision):
                recon, pred = model(input)
                pre_loss = loss_fn(pred, target)
                recon_loss = loss_fn(recon[:, :, -pre_dim:], input[:, :, -pre_dim:])
                loss_term_1 += pre_loss.item()
                loss_term_2 += recon_loss.item()
                loss = alpha_1 * pre_loss + alpha_2 * recon_loss

            scaler.scale(loss).backward()  # Scale the loss value
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
        avg_train_loss_term_1 = loss_term_1 / len(train_dataloader)
        avg_train_loss_term_2 = loss_term_2 / len(train_dataloader)
        total_train_loss = total_loss / len(train_dataloader)
        print('Epoch {},Total_train Loss: {:.4f}, pre_loss:{:.4f}, recon_loss:{:.4f}' \
              .format(epoch+1, total_train_loss, avg_train_loss_term_1, avg_train_loss_term_2))
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            for batch_idx, (input, target) in enumerate(val_dataloader):
                with autocast(enabled=args.use_mixed_precision):
                    input, target = input.to(args.device), target.to(args.device) # move data to device: cpu/gpu
                    _, pred = model(input)
                    pred = pred.cpu().numpy() 
                    target = target.cpu().numpy()
                    loss = loss_fn(torch.from_numpy(pred[:,:,-1]), torch.from_numpy(target[:,:,-1]))
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_dataloader)
            early_stopping(avg_val_loss, model,path=check_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer,epoch + 1, lradj=args.adj_lr_type,learning_rate=args.lr)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss       
   
                
    print('loading best model....')
    best_model_path = check_path+ '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path, map_location=args.device))
    model.eval()
    if os.path.exists(best_model_path):
            os.remove(best_model_path)
            print('Model weights deleted.')
    with torch.no_grad():
        test_pred = []
        test_target = []
        for batch_idx,(input, target) in enumerate(test_dataloader):
            with autocast(enabled=args.use_mixed_precision):
                input, target = input.to(args.device), target.to(args.device) 
                _, pred = model(input)              
                pred_copy = pred[:,:,-1].cpu().numpy() 
                target_copy = target[:,:,-1].cpu().numpy()  
                test_pred.append(pred_copy)
                test_target.append(target_copy) 
                if batch_idx % 1 == 0:
                        cur_input = input[0, :, -1].cpu().numpy()
                        cur_pred = pred[0, :, -1].cpu().numpy()
                        cur_target = target[0, :, -1].cpu().numpy()
                        pd=np.concatenate((cur_input, cur_pred))
                        gt=np.concatenate((cur_input, cur_target))
                        # plot predcition performance
                        visual(gt, pd, os.path.join(visual_folder, f'index '+str(batch_idx)+ '.pdf'))

        test_pred = np.concatenate(test_pred,axis=0)
        test_target = np.concatenate(test_target,axis=0)
        print('test shape:', test_pred.shape, test_target.shape)
        mae, mse, rmse, mape, mspe,_,_ =metric(test_pred,test_target)
        
        print('mse:{:.4f}, mae:{:.4f}'.format(mse, mae))
        # save metric to dict
        test_results= {
            'mse': float(round(mse, 4)),
            'mae': float(round(mae, 4)),
            'rmse':float(round(rmse,4)),
            'mape':float(round(mape,4)),
            'mspe':float(round(mspe,4)),
            }
    filename = f'.json'
    with open(os.path.join(save_folder, filename), 'w') as f:
        json.dump(test_results, f)
    