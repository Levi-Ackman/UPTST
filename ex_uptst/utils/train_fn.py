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
    train_seed,model, train_dataloader, val_dataloader,test_dataloader, device,pre_dim=7,lr=1e-4, alpha_list=[0.5, 0.5], criterion='mae',
    epochs=10, patience=5, batch_size=64, seq_len=384, pre_len=384,adj_lr_type='7',
    use_mixed_precision=True):
    alpha_1, alpha_2 = alpha_list
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    best_val_loss = float('inf')
    print('\n Training Start!')
    print("Model total parameters: {:.2f} M".format(sum(p.numel() for p in model.parameters())/1e+6))
    early_stopping = EarlyStopping(patience=patience, verbose=True,save=True)
    scaler = GradScaler(enabled=use_mixed_precision)  # Create a GradScaler object for mixed precision training
    
    check_head='/root/autodl-tmp/check_folder'+f'/ili_uptst/{train_seed}/{seq_len}->{pre_len}/a1_{alpha_1}/'
    check_tail='/'+criterion+'/'+f'batch_size_{batch_size}/'+f'lr_{lr}'
    visual_head=f'./visual/{train_seed}/{seq_len}->{pre_len}/a1_{alpha_1}/'
    visual_tail=check_tail
    dict_head=f'./test_dict/{train_seed}/{seq_len}->{pre_len}/a1_{alpha_1}/'
    dict_tail='/'+criterion+'/'+f'batch_size_{batch_size}/lr_{lr}/'
    
    check_path=check_head+check_tail
    visual_folder=visual_head+visual_tail
    save_folder=dict_head+dict_tail
    
        
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(check_path):
        os.makedirs(check_path)
    if not os.path.exists(visual_folder):
        os.makedirs(visual_folder)
    if criterion=='mse':
        loss_fn=nn.MSELoss()
    elif criterion=='mae':
        loss_fn=nn.L1Loss()
    for epoch in range(epochs):
        total_loss = 0.0
        loss_term_1 = 0.0
        loss_term_2 = 0.0
        model=model.to(device)
        model.train()
        for batch_idx, (input, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            # Move data to GPU
            input, target = input.to(device), target.to(device)
            with autocast(enabled=use_mixed_precision):
                recon, pred = model(input)
                pre_loss = loss_fn(pred, target)
                recon_loss = loss_fn(recon[:, :, -pre_dim:], input[:, :, -pre_dim:])
                loss_term_1 += pre_loss.item()
                loss_term_2 += recon_loss.item()
                alpha_1, alpha_2 = alpha_list
                loss = alpha_1 * pre_loss + alpha_2 * recon_loss

            scaler.scale(loss).backward()  # Scale the loss value
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
        avg_train_loss_term_1 = loss_term_1 / len(train_dataloader)
        avg_train_loss_term_2 = loss_term_2 / len(train_dataloader)
        total_train_loss = total_loss / len(train_dataloader)
        # print('Epoch {},Total_train Loss: {:.4f}, pre_loss:{:.4f}, recon_loss:{:.4f}' \
        #       .format(epoch+1, total_train_loss, avg_train_loss_term_1, avg_train_loss_term_2))
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            for batch_idx, (input, target) in enumerate(val_dataloader):
                with autocast(enabled=use_mixed_precision):
                    input, target = input.to(device), target.to(device) # 将数据移到GPU上处理
                    _, pred = model(input)
                    pred = pred.cpu().numpy() # 取预测值作为预测
                    target = target.cpu().numpy() # 取目标值作为真实值
                    loss = loss_fn(torch.from_numpy(pred[:,:,-pre_dim:]), torch.from_numpy(target[:,:,-pre_dim:]))
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_dataloader)
            early_stopping(avg_val_loss, model,path=check_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer,epoch + 1, lradj=adj_lr_type,learning_rate=lr)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss       
        with torch.no_grad():
            test_pred = []
            test_target = []
            for batch_idx, (input, target) in enumerate(test_dataloader):
                with autocast(enabled=use_mixed_precision):
                    input, target = input.to(device), target.to(device) # 将数据移到GPU上处理
                    _, pred = model(input)
                    pred_copy = pred[:,:,-pre_dim:].cpu().numpy() # 取预测值作为预测
                    target_copy = target[:,:,-pre_dim:].cpu().numpy() # 取目标值作为真实值
                    test_pred.append(pred_copy)
                    test_target.append(target_copy) 
        test_pred = np.concatenate(test_pred,axis=0)
        test_target = np.concatenate(test_target,axis=0)
        mae, mse, rmse, mape, mspe,_,_ =metric(test_pred,test_target)
        
        print(f'val_loss: {avg_val_loss:.4f} test_mae_loss: {mae:.4f}  test_mse_loss: {mse:.4f}')
                
    print('loading best model....')
    best_model_path = check_path+ '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        test_pred = []
        test_target = []
        for batch_idx,(input, target) in enumerate(test_dataloader):
            with autocast(enabled=use_mixed_precision):
                input, target = input.to(device), target.to(device) # 将数据移到GPU上处理
                model = model.to(device) # 将模型移到GPU上处理
                _, pred = model(input)
                pred_copy = pred[:,:,-pre_dim:].cpu().numpy() # 取预测值作为预测
                target_copy = target[:,:,-pre_dim:].cpu().numpy() # 取目标值作为真实值
                test_pred.append(pred_copy)
                test_target.append(target_copy) 
                if batch_idx % 1 == 0:
                        cur_input = input[0, :, -1].cpu().numpy()
                        cur_pred = pred[0, :, -1].cpu().numpy()
                        cur_target = target[0, :, -1].cpu().numpy()
                        pd=np.concatenate((cur_input, cur_pred))
                        gt=np.concatenate((cur_input, cur_target))
                        visual(gt, pd, os.path.join(visual_folder, f'index '+str(batch_idx)+ '.pdf'))
        test_pred = np.concatenate(test_pred,axis=0)
        test_target = np.concatenate(test_target,axis=0)
        print('test shape:', test_pred.shape, test_target.shape)
        mae, mse, rmse, mape, mspe,_,_ =metric(test_pred,test_target)
        
        print('mse:{:.4f}, mae:{:.4f}'.format(mse, mae))
        # 将 MAE、MSE 和 RMSE 值存储到字典中
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
    if os.path.exists(best_model_path):
            os.remove(best_model_path)
            print('Model weights deleted.')