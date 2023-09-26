import argparse
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer')

    # basic config
    parser.add_argument('--model', type=str, required=False, default='Autoformer',
                        help='model name, options: [DLinear,LSTM, Crossformer, FEDformer,PatchTST, Autoformer......]')
    parser.add_argument("--checkpoint_path", type=str, default='./check_folder',help="checkpoint_path")
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast]')
    
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')

    # data loader
    parser.add_argument("--data_path", type=str,default='/home/Paradise/Code_UPTST/dataset', help="Path to data folder")
    parser.add_argument("--data_name", type=str,default='cq_hfmd_2015_21.csv', help="data name")
    parser.add_argument("--tasks", type=str,default='MS_2', help="choose tasks from: [ \
         S: univariate,   MS_1: multi to single (w/o ano), MS_2: multi to single (with ano)]")
    parser.add_argument("--scale", type=bool,default=True, help="scale data with train_set")
    parser.add_argument("--split_ratio", type=list, default=[0.7,0.2,0.1], help="split_ratio of data (train_test_val)")
    
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='/root/autodl-tmp/check_folder/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=14, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=8, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout')
    parser.add_argument('--embed', type=str, default='learned',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--seed', type=int, default=2021, help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=2488, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='7', help='adjust learning rate')
    parser.add_argument('--use_amp',  type=bool, default=True, help='use automatic mixed precision training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer_op')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()
    if args.model=='Autoformer'or args.model=='FEDformer' :
        args.use_amp=False
    args.use_gpu = torch.cuda.is_available() and args.use_gpu

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    Exp = Exp_Long_Term_Forecast

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.learning_rate=1e-4 if args.batch_size==32 else 1e-3
    args.label_len=int(args.seq_len//2)
    print(f'\n>>>>> training with: model: {args.model} tasks:{args.tasks} seq_pre: {args.seq_len}->{args.pred_len} loss_fn: {args.loss} seed: {args.seed} lr: {args.learning_rate} bs: {args.batch_size} ')

    exp = Exp(args)
    exp.train()
    time_now = time.time()
    exp.test(test=1)
    print('Inference time:', time.time() - time_now)

    torch.cuda.empty_cache()
