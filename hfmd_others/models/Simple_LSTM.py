import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.lstm = nn.LSTM(input_size=configs.enc_in, hidden_size=configs.d_model, num_layers=configs.e_layers, batch_first=True)
        self.fc = nn.Linear(configs.d_model, configs.pred_len * configs.enc_in)
        self.dropout = nn.Dropout(configs.dropout)
        self.layernorm = nn.LayerNorm(normalized_shape=[configs.seq_len, configs.enc_in])
        self.channels=configs.enc_in
        
    def forward(self,x_enc, x_mark_enc, x_dec, x_mark_dec):
        B,T,N=x_enc.shape
        x_enc = self.layernorm(x_enc)
        lstm_out, _ = self.lstm(x_enc)
        lstm_out = self.dropout(lstm_out)
        
        predictions = self.fc(lstm_out[:,-1,:])
        
        pre=predictions.reshape(B, -1, self.channels)

        return pre

