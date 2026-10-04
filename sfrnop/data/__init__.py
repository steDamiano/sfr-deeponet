# ==============================================================================
# Portions of this code are adapted from sound-field-neural-network by Francesc Lluís Salvadó:
# https://github.com/francesclluis/sound-field-neural-network
#
# Original Copyright (c) 2020 Francesc Lluís Salvadó 
# Licensed under MIT License

# Copyright (c) 2020 Francesc Lluís Salvadó

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

from functools import partial
import numpy as np
from omegaconf import DictConfig

from torch import Generator, randperm
from torch.utils.data import DataLoader, Dataset, Subset
from hydra.utils import instantiate
# from core.praser import init_obj
import torch
import random
          
def define_dataloader(logger, config: DictConfig, phase: str = 'train'):
    """ create train/test dataloader and validation dataloader,  validation dataloader is None when phase is test or not GPU 0 """
    '''create dataset and set random seed'''
    phase_dataset, val_dataset = define_dataset(logger, config, phase)
    
    ''' create dataloader and validation dataloader '''
    dataloader = DataLoader(
        phase_dataset,
        batch_size= config.batch_size,
        num_workers= config.num_workers,
        shuffle= True if phase=='train' else False,
        pin_memory= True,
        drop_last= True if phase=='train' else False,
    )
    
    ''' val_dataloader don't use DistributedSampler to run only GPU 0! '''
    
    # dataloader_args.update(opt['datasets'][opt['phase']]['dataloader'].get('val_args',{}))
    val_dataloader = None
    if phase == 'train':
        val_dataloader = DataLoader(
            val_dataset,
            batch_size= config.val_batch_size,
            num_workers= config.num_workers,
            shuffle= False,
            pin_memory= True,
            drop_last= False 
        )
    return dataloader, val_dataloader


def define_dataset(logger, config: DictConfig, phase: str = 'train'):
    ''' loading Dataset() class from given file's name '''

    # phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')
    phase_dataset = instantiate(config.dataset)
    
    val_dataset = None

    valid_len = 0
    data_len = len(phase_dataset)
    
    # dataloder_opt = opt['datasets'][opt['phase']]['dataloader']
    valid_split = config.get('val_split', 0)    
    
    ''' divide validation dataset, valid_split==0 when phase is test or validation_split is 0. '''
    if valid_split > 0.0: 
        if isinstance(valid_split, int):
            assert valid_split < data_len, "Validation set size is configured to be larger than entire dataset."
            valid_len = valid_split
        else:
            valid_len = int(data_len * valid_split)
        
        data_len -= valid_len
        phase_dataset, val_dataset = subset_split(dataset=phase_dataset, lengths=[data_len, valid_len], generator=Generator().manual_seed(config.seed))
    
    logger.info('Number of {} samples: {}'.format(phase, data_len))
    
    if valid_len > 0:
        logger.info('Number of val samples: {}'.format(valid_len))   
    return phase_dataset, val_dataset

def subset_split(dataset, lengths, generator):
    """
    split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))
    return Subsets