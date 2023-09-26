import os
import torch
from models import Autoformer, DLinear, FEDformer, Simple_LSTM, PatchTST, Crossformer
from data_provider.data_factory import get_data

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'PatchTST': PatchTST,
            'LSTM': Simple_LSTM,
            'Crossformer': Crossformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.train_loader ,self.valid_loader, self.test_loader =get_data(data_path=args.data_path,data_name=args.data_name,
                freq=args.freq,scale=args.scale,seq_len=args.seq_len,pre_len=args.pred_len,batch_size=args.batch_size,
                split_ratio=args.split_ratio,tasks=args.tasks,
                )

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
