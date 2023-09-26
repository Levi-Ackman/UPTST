import sys
sys.path.append('..')
from my_model.patchTST import PatchTST
import torch.nn as nn
import torch
from utils.tools import create_patch
class U_PatctTST(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x patch_len x n_vars x num_patch] for pretrain
    """
    def __init__(
        self, 
        in_channel:int,          ## n_vars
        pre_dim:int,             ## target_feature_dims to predicted
        pre_len:int,             ## final out_dim(pre_len)
        seq_len:int,             ## input_dim (seq_len)
        patch_len:int,           ## patch_len
        
        n_layers:int=3,          ## n_layers U-net layers
        expansion=4,             ## expasion rate for U-net
        PacthTST_Dep:int=2,      ## n_layers of PatchTST  block
        d_model=256,             ## model_dim of PatchTST block
        n_heads=8,               ## num_heads of PatchTST block
        shared_embedding=True,   ## shared_embedding for PatchTST block
        d_ff:int=1024,            ## feedforward_dim of PatchTST block
        norm_type:str='bn',      ## norm_type of ConvNet
        attn_dropout:float=0.,     ## attn_dropout of PatchTST block
        dropout:float=0.,          ## dropout rate of PatchTST block
        conv_dropout:float=0.,          ## dropout rate of Conv block
        act_type:str="gelu",             ## act_type of Conv
        res_attention:bool=True, 
        pre_norm:bool=True,        ## for PatchTST block
        store_attn:bool=False,      ## for PatchTST block
        pe:str='sincos',             ## position encoding for PatchTST block
        learn_pe:bool=True,         ## position encoding for PatchTST block
        head_dropout = 0.2,           ## head_dropout for PatchTST block
        individual = False,         ## channel individual/independent for PatchTST block
        revin=False,
        ):

        super().__init__()
        ori_len=int(seq_len)
        seq_lens=[]
        seq_lens.append(ori_len)
        num_patchs=[]
        for _ in range(n_layers):
            if ori_len % patch_len == 0:
                num_patch = int((ori_len - patch_len) // patch_len + 1)
                num_patchs.append(num_patch)
            else:
                num_patch = int((ori_len - patch_len) // (patch_len - 1) + 1)
                num_patchs.append(num_patch)
            ori_len/=2
            seq_lens.append(ori_len)
        
        self.n_layers = n_layers
        self.patch_len=patch_len
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.down_patchtst_blocks= nn.ModuleList()
        self.up_patchtst_blocks= nn.ModuleList()
        self.seq_len=seq_len
        self.pre_len=pre_len
        self.pre_dim=pre_dim
        self.revin=revin
        
        for i in range(n_layers):
            self.down_blocks.append(nn.Sequential(
                DownConvBlock(in_channel, int(in_channel*expansion),dropout=conv_dropout,act_type=act_type,norm_type=norm_type),
                DownConvBlock(int(in_channel*expansion), in_channel,dropout=conv_dropout,act_type=act_type,norm_type=norm_type,kernel_size=3,stride=1,padding=1),
            ))
            self.down_patchtst_blocks.append(PatchTST(c_in=in_channel,target_dim=int(seq_lens[i]), patch_len=patch_len, num_patch=int(num_patchs[i]), 
                                n_layers=PacthTST_Dep, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                head_dropout=head_dropout,individual=individual,head_type='prediction',
                                pe=pe, learn_pe=learn_pe))
        for i in range(n_layers):
            self.up_blocks.append(nn.Sequential(
                UpConvBlock(in_channel, int(in_channel*expansion),dropout=conv_dropout,act_type=act_type,norm_type=norm_type),
                UpConvBlock(int(in_channel*expansion), in_channel,dropout=conv_dropout,act_type=act_type,norm_type=norm_type,kernel_size=3,stride=1,padding=1)
            ))
            if i !=n_layers-1:
                self.up_patchtst_blocks.append(PatchTST(c_in=in_channel,target_dim=int(seq_lens[n_layers-i-1]), patch_len=patch_len, num_patch=int(num_patchs[n_layers-i-1]), 
                                n_layers=PacthTST_Dep, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                head_dropout=head_dropout,individual=individual,head_type='prediction',
                                pe=pe, learn_pe=learn_pe))
            else:
                self.up_patchtst_blocks.append(PatchTST(c_in=in_channel,target_dim=int(seq_lens[n_layers-i-1]+pre_len), patch_len=patch_len, num_patch=int(num_patchs[n_layers-i-1]), 
                                n_layers=PacthTST_Dep, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                head_dropout=head_dropout,individual=individual,head_type='prediction',
                                pe=pe, learn_pe=learn_pe))
                
    def forward(self,x):
        if self.revin:
            # x: [bs x seq_len x nvars]
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x =x/ stdev
        down_features=[] # record down-feature for residual connection
        for i in range(self.n_layers):
            x,_=create_patch(x,patch_len=self.patch_len,stride=self.patch_len)
            x=self.down_patchtst_blocks[i](x)
            down_features.append(x)
            
            x=x.permute(0,2,1)
            x=self.down_blocks[i](x)
            x=x.permute(0,2,1)
            
            
        for i in range(self.n_layers):
            x=x.permute(0,2,1)
            x=self.up_blocks[i](x)
            x=x.permute(0,2,1)
            _,valid_len,_=down_features[self.n_layers-i-1].shape
            x=x[:,:valid_len,:]+down_features[self.n_layers-i-1]
            x,_=create_patch(x,patch_len=self.patch_len,stride=self.patch_len)
            x=self.up_patchtst_blocks[i](x)
         # De-Normalization from Non-stationary Transformer
        if self.revin:
            x = x * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len+self.pre_len, 1))
            x = x + \
                    (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len+self.pre_len, 1))
        recon=x[:,:self.seq_len,:]
        pre=x[:,self.seq_len:,:]
        
        return recon[:,:,-self.pre_dim:],pre[:,:,-self.pre_dim:]

        
class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout,norm_type='bn',act_type='glue',kernel_size=3, stride=2, padding=1):
        super(DownConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dropout=nn.Dropout(p=dropout)
        if norm_type =='bn':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.InstanceNorm1d(out_channels)
        if act_type == 'glue':
            self.act =nn.GELU()
        elif act_type =='leaky':
            self.act =nn.LeakyReLU(inplace=True)
        else:
            self.act =nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x=self.dropout(x)
        x = self.act(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout,norm_type='bn',act_type='glue',kernel_size=2, stride=2, padding=0):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dropout=nn.Dropout(p=dropout)
        if norm_type =='bn':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.InstanceNorm1d(out_channels)
        if act_type == 'glue':
            self.act =nn.GELU()
        elif act_type =='leaky':
            self.act =nn.LeakyReLU(inplace=True)
        else:
            self.act =nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv(x)
        x = self.norm(x)
        x=self.dropout(x)
        x = self.act(x)
        return x

if __name__ == "__main__":
    # input: tensor [bs x seq_len x n_vars]
    bs, nvars, seq_len,pre_len= 1,8,192,384
    target_feature_dim=2
    
    xb = torch.randn(bs,seq_len,nvars)
    model=U_PatctTST(
        in_channel=nvars,
        pre_dim=target_feature_dim,
        pre_len=pre_len,
        seq_len=seq_len,
        patch_len=8,
    )
    recon,pre=model(xb)
    print(f'input shape:{xb.shape}]')
    print(f'output shape:{pre.shape}]')
