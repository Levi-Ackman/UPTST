import sys
sys.path.append('..')
from my_model.layers.encoders import PatchTSTEncoder
from my_model.layers.heads import *

            
# Cell
class PatchTST(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction /here target_dim represents prediction/forecasting_len
         [bs x target_dim] for classification /here target_dim represents number of classes
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, num_patch:int, 
                 n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "prediction", individual = False, 
                verbose:bool=False, **kwargs):

        super().__init__()

        assert head_type in ['prediction', 'classification'], 'head type should be either prediction, or classification'
        # Backbone
        self.backbone = PatchTSTEncoder(c_in, num_patch=num_patch, patch_len=patch_len, 
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,norm=norm,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.n_vars = c_in
        self.head_type = head_type
        
        if head_type == "prediction":
            self.head = LinearPredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)
        elif head_type == "classification":
            self.head = LinearClassificationHead(self.n_vars, d_model, target_dim, head_dropout)


    def forward(self, z):                             
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """   
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x num_patch]
        z = self.head(z)                                                                    
        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for classification
        return z

if __name__ == "__main__":
    # z: tensor [bs x num_patch x n_vars x patch_len]
    bs, num_patch, nvars, patch_len= 1,4,8,24
    pre_len=num_patch*patch_len
    
    xb = torch.randn(bs, num_patch, nvars, patch_len)
    model=PatchTST(
        c_in=nvars,
        target_dim=pre_len,
        patch_len=patch_len,
        num_patch=num_patch,
        )
    y=model(xb)
    print(f'input shape:{xb.shape}]')
    print(f'output shape:{y.shape}]')

