from torch.utils.data import Dataset
from conv_tasnet import TasNet
import sdr
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import pickle
import numpy as np
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import os 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
from time import time
import multiprocessing as mp

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cuda = True
use_cuda = cuda and torch.cuda.is_available()
# Handel GPU stochasticity
torch.backends.cudnn.enabled = use_cuda
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if use_cuda else "cpu")
def to_device(dicts, device):
    '''
       load dict data to cuda
    '''
    def to_cuda(datas):
        if isinstance(datas, torch.Tensor):
            return datas.to(device)
        elif isinstance(datas,list):
            return [data.to(device) for data in datas]
        else:
            raise RuntimeError('datas is not torch.Tensor and list type')

    if isinstance(dicts, dict):
        return {key: to_cuda(dicts[key]) for key in dicts}
    else:
        raise RuntimeError('input egs\'s type is not dict')

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)



class New_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.recording_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
   
        return self.recording_df.shape[0]



    def __getitem__(self, idx):

        idx = idx 
        if torch.is_tensor(idx):
            idx = idx.tolist()
            

        record_path = self.recording_df.loc[idx, "path_file"]
        #record_path = record_path[0:15]+'1'+record_path[15:]
        with open(record_path, "rb") as f:
           
            _, noisy_signal, _, speakers_target, _ = pickle.load(f)
            mixed_sig_np = noisy_signal-np.mean(noisy_signal)
            speakers_target[0] -= np.mean(speakers_target[0])
            speakers_target[1] -= np.mean(speakers_target[1])


        sample_separation = {'mixed_signals': mixed_sig_np, 'clean_speeches': speakers_target}
        
        
        return sample_separation

def try_test(net_path, test_path, device):
    
    nnet = TasNet().to(device)
    nnet.load_state_dict(torch.load(net_path))
    data = New_dataset(test_path)
    datloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8, collate_fn=default_collate, pin_memory=True)
    datloader = DeviceDataLoader(datloader,device)
    for i, data in enumerate(datloader, 0):
      if i > 3:
          break
      input = data['mixed_signals']
      labels = data['clean_speeches']
      #outputs=torch.nn.functional.normalize(nnet(input),p=1)
      outputs=nnet(input)
      loss = -sdr.batch_SDR_torch(outputs, labels)
      loss = sum(loss)/labels.shape[0]
      loss2 = -sdr.batch_SDR_torch(labels, labels)
      print(loss)
      print(loss2)
      a=input.detach().cpu().numpy()
      a.shape =a.shape[1]
      print(a.shape)
      clean1 = labels[0][0].detach().cpu().numpy()
      clean2 = labels[0][1].detach().cpu().numpy()
      print(clean1.dtype)
      out1= outputs[0][0].detach().cpu().numpy()
      out1 = out1/max(np.abs(out1))
      out2= outputs[0][1].detach().cpu().numpy()
      out2 = out2/max(np.abs(out2))
      print(out1.dtype)
      filename1 = str(i)+'Speaker1.wav'
      filename2 = str(i)+'Speaker2.wav'
      write(filename1, 16000,out1)
      write(filename2, 16000,out2)
      write('cl'+filename1, 16000,clean1)
      write('cl'+filename2, 16000,clean2)
      write(' mixed '+str(i)+'.wav', 16000, a) 

def test_loss(net_path, test_path, device):
    
    nnet = TasNet().to(device)
    nnet.load_state_dict(torch.load(net_path))
    data = New_dataset(test_path)
    datloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=8, collate_fn=default_collate, pin_memory=True)
    datloader = DeviceDataLoader(datloader,device)
    running_loss = 0
    for i, data in enumerate(datloader, 0):
      print(i)
      input = data['mixed_signals']
      labels = data['clean_speeches']
      outputs=nnet(input)
      loss = -sdr.batch_SDR_torch(outputs, labels).data
      #loss = sum(loss)/labels.shape[0]
      running_loss+= loss
    print(running_loss/i)     
    
   



if __name__ == '__main__':
    
    data = New_dataset("/dsi/gannot-lab1/datasets/mordehay/data_wham_partial_overlap/train/csv_files/with_wham_noise_res_more_informative_correct.csv") 
   
    #indx = np.arange(500)
    #datloader = DataLoader(data, batch_size=16, shuffle=False, sampler=SubsetRandomSampler(indx), collate_fn=default_collate)
    
    '''
    dur = 10000000
    for num_workers in range(2, mp.cpu_count()+1, 2):  
        data = New_dataset("/home/dsi/rinav/Conv-TasNet/try.csv")
        train_loader = DataLoader(data,shuffle=True,num_workers=num_workers,batch_size=8, collate_fn=default_collate, pin_memory=True)
        #train_loader = DeviceDataLoader(train_loader,device)
        print('Start clacultaions')
        start = time()
        for i, data in enumerate(train_loader, 0):
            if (i*8) >= 1000:
                break
        end = time()
        duration = end-start
        print("num worker is:{}, duration time is:{}".format(num_workers, duration))
        if duration<dur:
            dur = duration
            nw = num_workers
    print("num_workwers is:{}".format(nw))
    '''
    datloader = DataLoader(data, batch_size=16, shuffle=True, num_workers=8, collate_fn=default_collate, pin_memory=True)
    datloader = DeviceDataLoader(datloader,device)
    nnet = TasNet().cuda().to(device)
    optimizer = optim.Adam(nnet.parameters(), lr=0.001)
    train_loss = []

    for epoch in range(100):  # loop over the dataset multiple times
        print(epoch)

        running_loss = 0.0
        for i, data in enumerate(datloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['mixed_signals'], data['clean_speeches']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = nnet(inputs)
            #print(outputs.shape)
            loss = -sdr.batch_SDR_torch(outputs, labels)
            loss = sum(loss)/labels.shape[0]
            running_loss+= loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nnet.parameters(),1)
            optimizer.step()
        
        train_loss.append(running_loss/i)
        print(train_loss[epoch])
        if epoch%10 ==0:
          PATH = './nnet_epoch{}.pth'.format(epoch)
          torch.save(nnet.state_dict(), PATH)
    print('Finished Training')
    PATH = './try_nnet.pth'
    torch.save(nnet.state_dict(), PATH)
   
    # plot loss on training set
    steps = np.arange(100)
    fig, ax = plt.subplots()
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title('Train Loss')
    ax.plot(steps, train_loss.detach().cpu().numpy(), label="train loss", color='red')
    fig.legend(loc='center',bbox_to_anchor=(0.5, 0.3, 0.5, 0.5))
    fig.tight_layout()
    plt.show()

    print('Finished Training')
    PATH = './try_nnet.pth'
    torch.save(nnet.state_dict(), PATH)
    
    test_path = "/dsi/gannot-lab1/datasets/mordehay/data_wham_partial_overlap/test/csv_files/with_wham_noise_res_more_informative_correct.csv"
    
    try_test(PATH,test_path, device)