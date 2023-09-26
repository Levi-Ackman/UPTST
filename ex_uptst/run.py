import random
import torch
import numpy as np
import argparse
from utils.train_wrapper import train_model_wrapper,get_model_params

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(device))
    parser = argparse.ArgumentParser(description="Training UPTST wrapper")
    parser.add_argument("--checkpoint_path", type=str, default='./check_folder',help="checkpoint_path")
    parser.add_argument("--data_path", type=str,default='/home/Paradise/UPTST/dataset/', help="Path to data folder")
    parser.add_argument("--data_name", type=str,default='exchange_rate.csv', help="data name")
    parser.add_argument("--seq_len", type=int, default=24, help="Sequence length")
    parser.add_argument("--pre_len", type=int, default=24, help="prediction length")
    parser.add_argument("--bs", type=int,default=32, help="Batch size")
    parser.add_argument("--scale", type=bool,default=True, help="scale data with train_set")
    parser.add_argument("--split_ratio", type=list, default=[0.7,0.2,0.1], help="split_ratio of data (train_test_val)")
    parser.add_argument("--patch_len", type=int,default=32, help="Number of patches")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--lr", type=float, default=1e-3,help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha values of loss term")
    parser.add_argument("--loss_fn", type=str,default='mse', help="Loss function")
    parser.add_argument("--epochs", type=int, default=2408,help="Number of epochs")
    parser.add_argument("--patience", type=int, default=3,help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=2021,help="training_seed")
    

    # Model parameters
    parser.add_argument("--n_layers", type=int, default=2, help="Number of U-net block layers")
    parser.add_argument("--expansion", type=int, default=4, help="Expansion rate for U-net")
    parser.add_argument("--PacthTST_Dep", type=int, default=2, help="Number of PatchTST layers")
    parser.add_argument("--d_model", type=int, default=256 , help="Model dimension for PatchTST block")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads for PatchTST block")
    parser.add_argument("--shared_embedding",type=bool,default=True, help="Use shared embedding for PatchTST block")
    parser.add_argument("--d_ff", type=int, default=1024 , help="Feedforward dimension for PatchTST block")
    parser.add_argument("--norm_type", type=str, default="bn", help="Normalization type for ConvNet")
    parser.add_argument("--attn_dropout", type=float, default=0.3, help="Attention dropout rate for PatchTST block")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for PatchTST block")
    parser.add_argument("--conv_dropout", type=float, default=0.2, help="Dropout rate for Conv block")
    parser.add_argument("--act_type", type=str, default="gelu", help="Activation type for Conv block")
    parser.add_argument("--res_attention", type=bool,default=False, help="Enable residual attention")
    parser.add_argument("--pre_norm", type=bool,default=True, help="Enable pre-normalization for PatchTST block")
    parser.add_argument("--store_attn", type=bool,default=False, help="Enable storing attention maps")
    parser.add_argument("--pe", type=str, default="sincos", help="Position encoding for PatchTST block,choose from [zero,zeros,normal/gauss,uniform,sincos]")
    parser.add_argument("--learn_pe",type=bool,default=True, help="Enable learning position encoding")
    parser.add_argument("--head_dropout", type=float, default=0.3, help="Head dropout rate for PatchTST block")
    parser.add_argument("--individual", type=bool,default=False, help="Enable channel individual/independent for final prediction head")


    args = parser.parse_args()
    args.model='UPTST'
    if args.seq_len ==96 :args.patch_len = 48
    args.lr=2.5e-4 if args.bs==32  else 1e-4
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f'\n>>>>> ILI: training with: model: {args.model} seq_pre: {args.seq_len}->{args.pre_len} loss_fn: {args.loss_fn} seed: {args.seed} alpha_1 {args.alpha} lr: {args.lr} bs: {args.bs} ')
    train_model_wrapper(
        train_seed=args.seed,
        data_path=args.data_path,
        data_name=args.data_name,
        seq_pre=[args.seq_len,args.pre_len],
        bs=args.bs,
        patch_len=args.patch_len,
        device=args.device,
        lr=args.lr,
        alpha_list=[args.alpha,1.0-args.alpha],
        loss_fn=args.loss_fn,
        epochs=args.epochs,
        patience=args.patience,
        scale=args.scale,
        split_ratio=args.split_ratio,
        model_params=get_model_params(args)
                )
    
    torch.cuda.empty_cache()