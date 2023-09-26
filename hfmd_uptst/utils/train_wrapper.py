from my_model.U_PatctTST import U_PatctTST
import torch
from data_provider.get_dataloader import get_data
from utils.train_fn import train_model
def train_model_wrapper(args):
    train_loader, val_loader, test_loader = get_data(
        data_path=args.data_path,
        data_name=args.data_name,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        batch_size=args.bs,
        tasks=args.tasks,
        scale=args.scale,
        split_ratio=args.split_ratio,
    )
    x, y = next(iter(train_loader))
    _,seq_len,n_vars = x.shape
    _,pre_len,target_dim = y.shape
    print(f'seq_len: {seq_len}, pre_len: {pre_len} input_dim: {n_vars} target_dim: {target_dim}')

    model = U_PatctTST(
        in_channel=n_vars,
        pre_dim=target_dim,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        patch_len=args.patch_len,
        **get_model_params(args)
    )

    train_model(
        args,
        model=model,
        pre_dim=target_dim,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
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