import sys
sys.path.append('..')
from data_provider.pred_dataset import hand_foot
from torch.utils.data import DataLoader

def get_data(data_path='/home/Paradise/UPTST/dataset',
             data_name='cq_hfmd_2015_21.csv',seq_len=384,pre_len=384,batch_size=64,
             split_ratio=[0.7,0.2,0.1],num_workers=0,scale=False,tasks='MS_1',
             ):
    dataset=hand_foot
    print(tasks)
    train_dataset=dataset(root_path=data_path,data_type='train', size=[seq_len,pre_len],split_ratio=split_ratio,
                  data_path=data_name,tasks=tasks,
                 scale=scale)
    valid_dataset=dataset(root_path=data_path,data_type='val',size=[seq_len,pre_len], split_ratio=split_ratio,
                  data_path=data_name,tasks=tasks,
                 scale=scale)
    test_dataset=dataset(root_path=data_path,data_type='test', size=[seq_len,pre_len],split_ratio=split_ratio,
                  data_path=data_name,tasks=tasks,
                 scale=scale)
    
    print('len(train_dataset): {}'.format(len(train_dataset)))
    print('len(valid_dataset): {}'.format(len(valid_dataset)))
    print('len(test_dataset): {}'.format(len(test_dataset)))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=True)
    val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False,num_workers=num_workers,drop_last=False)
    
    return train_loader,val_loader,test_loader

if __name__ == "__main__":
    train_loader,val_loader,test_loader=get_data(data_path='/home/Paradise/HFMD',
             data_name='cq_hfmd_2015_21.csv',seq_len=96,pre_len=96,batch_size=64,
             split_ratio=[0.7,0.2,0.1],num_workers=0,scale=False,tasks='MS_2',
             )
    for batch_idx, (input, target) in enumerate(test_loader):
        print(f"Batch {batch_idx}, Input shape: {input.shape}, Target shape: {target.shape}")
 
    
    