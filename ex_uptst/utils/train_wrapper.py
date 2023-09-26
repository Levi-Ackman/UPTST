from my_model.U_PatctTST import U_PatctTST
import torch
from data_provider.get_dataloader import get_data
from utils.train_fn import train_model
def train_model_wrapper(train_seed,data_path, data_name, seq_pre, bs, patch_len, device, lr, alpha_list, loss_fn,
                        epochs, patience,split_ratio,scale,model_params):
    train_loader, val_loader, test_loader = get_data(
        data_path=data_path,
        data_name=data_name,
        seq_len=seq_pre[0],
        pre_len=seq_pre[1],
        batch_size=bs,
        scale=scale,
        split_ratio=split_ratio,
    )
    x, y = next(iter(train_loader))
    _,seq_len,n_vars = x.shape
    _,pre_len,target_dim = y.shape
    print(f'seq_len: {seq_len}, pre_len: {pre_len} target_dim: {target_dim}')

    model = U_PatctTST(
        in_channel=n_vars,
        pre_dim=target_dim,
        pre_len=seq_pre[1],
        seq_len=seq_pre[0],
        patch_len=patch_len,
        **model_params
    )

    train_model(
        train_seed,
        model=model,
        pre_dim=target_dim,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        device=device,
        lr=lr,
        adj_lr_type='7',
        alpha_list=alpha_list,
        criterion=loss_fn,
        batch_size=bs,
        seq_len=seq_pre[0],
        pre_len=seq_pre[1],
        epochs=epochs,
        patience=patience,
    )

    torch.cuda.empty_cache()
def get_model_params(args):
    return {
        "n_layers": args.n_layers,
        "expansion": args.expansion,
        "PacthTST_Dep": args.PacthTST_Dep,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "shared_embedding": args.shared_embedding,
        "d_ff": args.d_ff,
        "norm_type": args.norm_type,
        "attn_dropout": args.attn_dropout,
        "dropout": args.dropout,
        "conv_dropout": args.conv_dropout,
        "act_type": args.act_type,
        "res_attention": args.res_attention,
        "pre_norm": args.pre_norm,
        "store_attn": args.store_attn,
        "pe": args.pe,
        "learn_pe": args.learn_pe,
        "head_dropout": args.head_dropout,
        "individual": args.individual,
    }