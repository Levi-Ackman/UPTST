from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if flag == 'train':
            return self.train_loader.dataset, self.train_loader
        if flag == 'val':
            return self.valid_loader.dataset, self.valid_loader
        if flag == 'test':
            return self.test_loader.dataset, self.test_loader

    def _select_optimizer(self):
        if self.args.optimizer == 'Adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'SGD':
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'RMSprop':
            model_optim = optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate)

        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MAE' or self.args.loss == 'mae':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def _vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, -1:]
                batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        _, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        check_head=f'{self.args.checkpoints}{self.args.seed}/{self.args.seq_len}->{self.args.pred_len}/'
        check_tail=f'/{self.args.model}/{self.args.loss}/bz_{self.args.batch_size}/lr_{self.args.learning_rate}'
        best_model_path=check_head+self.args.tasks+check_tail
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        outputs = outputs[:, -self.args.pred_len:, -1:]
                        batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = outputs[:, -self.args.pred_len:, -1:]
                    batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self._vali(vali_data, vali_loader, criterion)
            test_loss = self._vali(test_data, test_loader, criterion)

            print("Epoch: {}, Steps: {} | Train Loss: {:.4f} Vali Loss: {:.4f} Test Loss: {:.4f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, best_model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # self.model.load_state_dict(torch.load(best_model_path + '/' + 'checkpoint.pth'))

        # return self.model

    def test(self, test=1):
        _, test_loader = self._get_data(flag='test')

        check_head=f'{self.args.checkpoints}{self.args.seed}/{self.args.seq_len}->{self.args.pred_len}/'
        check_tail=f'/{self.args.model}/{self.args.loss}/bz_{self.args.batch_size}/lr_{self.args.learning_rate}/checkpoint.pth'
        best_model_path=check_head+self.args.tasks+check_tail

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
            print('Model weights deleted.')

        preds = []
        trues = []
        visual_head=f'./visual/{self.args.seed}/{self.args.seq_len}->{self.args.pred_len}/'
        visual_tail=f'/{self.args.model}/{self.args.loss}/bz_{self.args.batch_size}/lr_{self.args.learning_rate}'
        dict_head=f'./test_dict/{self.args.seed}/{self.args.seq_len}->{self.args.pred_len}/'
        dict_tail=f'/{self.args.model}/{self.args.loss}/bz_{self.args.batch_size}/lr_{self.args.learning_rate}'
        
        visual_folder=visual_head+self.args.tasks+visual_tail
        dict_folder=dict_head+self.args.tasks+dict_tail
        
        if not os.path.exists(visual_folder):
            os.makedirs(visual_folder)
        if not os.path.exists(dict_folder):
            os.makedirs(dict_folder)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                outputs = outputs[:, -self.args.pred_len:, -1:]
                batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
                if i % 24 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(visual_folder, str(i) + '.pdf'))
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{:.4f}, mae:{:.4f}'.format(mse, mae))
        my_dict = {
            'mse': float(round(mse, 4)),
            'mae': float(round(mae, 4)),
            'rmse':float(round(rmse,4)),
            'mape':float(round(mape,4)),
            'mspe':float(round(mspe,4)),
            }
        with open(os.path.join(dict_folder, 'records.json'), 'w') as f:
            json.dump(my_dict, f)
        f.close()
