import torch.utils.data as data
from torchvision import transforms
import os
import torch
import numpy as np
import random
import h5py
from sfrnop.data.mask import get_sfr_mask
import tqdm

def get_frequencies():
    freqs_path = 'data/util/frequencies.txt'
    with open(freqs_path) as f:
        freqs = [[int(freq) for freq in line.strip().split(' ')] for line in f.readlines()][0]
    return freqs


def scale(soundfield):
    for i in range(soundfield.shape[-1]):
        max_abs_freq_i = np.max(soundfield[:,:,i])
        soundfield[:,:,i] = soundfield[:,:,i]/max_abs_freq_i
    return soundfield

def scale_single_freq(soundfield):
    max_abs_freq = np.max(soundfield)
    soundfield = soundfield / max_abs_freq
    return soundfield
    
class SFRHDFDataset(data.Dataset):
    def __init__(self, data_root, num_samples=-1, freq=None, image_size=[32, 32]):

        self.freqs = get_frequencies()
        self.freq = freq
        
        self.tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.image_size = image_size

        grid_x, grid_y = torch.meshgrid((torch.arange(self.image_size[0]))/self.image_size[0], (torch.arange(self.image_size[1]))/self.image_size[0], indexing='ij')
        grid_positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).float()
        self.grid_positions = grid_positions

        self.data_root = data_root
        
        if num_samples > 0:
            self.length = num_samples
        else:
            with h5py.File(os.path.join(self.data_root, 'soundfield_data.h5'), 'r') as hf:
                self.length = hf['soundfield_data'].shape[0]
        
        self.h5_file = h5py.File(os.path.join(self.data_root, 'soundfield_data.h5'), 'r')
        
        # Load all data
        temp_data = self.h5_file['soundfield_data']
        self.data = torch.zeros(temp_data.shape)
        
        for i in tqdm.tqdm(range(self.length)):
            arr = temp_data[i]
            scaled_arr = self.tfs(scale(arr)).movedim(0,2)
            self.data[i] = scaled_arr
        
        # Load and decode all paths
        temp_paths = self.h5_file['file_paths']
        self.paths = list(temp_paths)

        self.max_dim = 0
        for i in range(len(self.paths)):
            self.paths[i] = self.paths[i].decode('utf-8').rsplit("/")[-1].rsplit("\\")[-1]
            cur_dimensions = self.paths[i].split('_')[2:4]
            cur_dimensions = [float(val) for val in cur_dimensions]
            if max(cur_dimensions) > self.max_dim:
                self.max_dim = max(cur_dimensions)

    def __getitem__(self, index):
        ret = {}
        
        if self.freq is not None:
            freq = self.freq
        else:
            freq = random.choice(self.freqs)

        freq_idx = self.freqs.index(freq)
        img = self.data[index][:,:,freq_idx]
        
        # Room size and source position
        file_split = self.paths[index].split('_')
        geometry_params = file_split[2:4] + file_split[6:8]
        
        # Convert to float and normalize to max room dimension
        geometry_params = [float(val)/self.max_dim for val in geometry_params]
        
        # Conditioning vector
        cond_vector = torch.cat((torch.tensor([freq / max(self.freqs)]), torch.tensor(geometry_params)), dim=0)
        
        # Original return values
        ret['gt_image'] = img.reshape(-1,1)
        ret['cond_params'] = cond_vector.unsqueeze(1) 
        ret['path'] = self.paths[index]

        ret['trunk_input'] = self.grid_positions
        ret['trunk_target'] = img.reshape(-1,1)

        ret['sensor_pos'] = self.grid_positions
        ret['sensor_val'] = img.reshape(-1,1)
        
        ret['temporal_frequency'] = freq
        ret['room_dimensions'] = torch.tensor(geometry_params[:2])
        
        return ret

    def __len__(self):
        return self.length

    def get_mask(self):
        if self.mask_mode == 'sfr_mask':
            mask = get_sfr_mask(self.image_size, random.randint(64, 512))
        elif self.mask_mode == 'sfr_mask_4':
            mask = get_sfr_mask(self.image_size, 4)
        elif self.mask_mode == 'sfr_mask_8':
            mask = get_sfr_mask(self.image_size, 8)
        elif self.mask_mode == 'sfr_mask_16':
            mask = get_sfr_mask(self.image_size, 16)
        elif self.mask_mode == 'sfr_mask_32':
            mask = get_sfr_mask(self.image_size, 32)
        elif self.mask_mode == 'sfr_mask_64':
            mask = get_sfr_mask(self.image_size, 64)
        elif self.mask_mode == 'sfr_mask_128':
            mask = get_sfr_mask(self.image_size, 128)
        elif self.mask_mode == 'sfr_mask_256':
            mask = get_sfr_mask(self.image_size, 256)
        elif self.mask_mode == 'sfr_mask_512':
            mask = get_sfr_mask(self.image_size, 512)
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask)
