import os
from sfrnop.train import get_pretrained_model_checkpoint
from lightning import seed_everything
import hydra
from dotenv import load_dotenv
from lightning import LightningModule, Trainer
from pathlib import Path
from omegaconf import DictConfig
import logging 
from sfrnop.data import define_dataloader
import numpy as np
import torch 
from sfrnop.models.setonet import get_batch_mask
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

def NCC(u_estimated: torch.Tensor, u_ground_truth: torch.Tensor) -> torch.Tensor:
    '''
    Normalized Cross-Correlation (NCC) between estimated and ground truth sound fields.
    
    Parameters
    ----------
    u_estimated: torch.Tensor
        Estimated sound field.
    u_ground_truth: torch.Tensor
        Ground truth sound field.
    
    Returns
    -------
    NCC: torch.Tensor
        NCC value.
    '''
    u_estimated = u_estimated.ravel()
    u_ground_truth = u_ground_truth.ravel()
    return torch.abs(u_estimated @ torch.conj(u_ground_truth)) / (torch.norm(u_estimated, p=2) * torch.norm(u_ground_truth, p=2))


def nmse_tot(input, target):
    output = 10*np.log10(np.linalg.norm(input - target)**2/(np.linalg.norm(target)**2))
    return output

@hydra.main(version_base=None, config_path="configs/", config_name="base")
def evaluate(config: DictConfig):
    log.info(f"Evaluating with configuration: {config}")

    eval_config = config.evaluation

    seed_everything(config.training.seed, workers=True)
    log.info(f"Seed: {config.training.seed}")

    # Load from checkpoint
    ckpt_path = get_pretrained_model_checkpoint(
        pretrained_model=eval_config.alias,
        work_folder=config.env.work_folder,
    )

    model: LightningModule = hydra.utils.get_class(config.model._target_).load_from_checkpoint(
        checkpoint_path=ckpt_path,
        **config.model
    )
    
    test_dataloader, _ = define_dataloader(log, eval_config, "test")

    test_mse = 0
    test_ncc = 0
    bn= 0 
    with torch.no_grad():
        for batch_data in test_dataloader:
         #  Sensor position on full grid
            sensor_pos = batch_data['trunk_input']

            # Target on full grid
            sensor_val = batch_data['gt_image']

            # Generate mask for entire batch
            batch_mask = get_batch_mask(sensor_pos.shape[0], n_mics=eval_config.num_microphones, mask_size=sensor_pos.shape[1])
            
            masked_indices = torch.nonzero((batch_mask)).reshape(-1,1)[:,0]
            sensor_pos = sensor_pos[:, masked_indices, :].to(model.device)
            sensor_val = sensor_val[:, masked_indices, :].to(model.device)
            branch_freq = batch_data['cond_params'].to(model.device)
            trunk_input = batch_data['trunk_input'].to(model.device)
            trunk_target = batch_data['trunk_target'].to(model.device)
            
            output = model(sensor_pos, sensor_val, branch_freq, trunk_input)
            
            if test_mse == 0:
                nmse = nmse_tot(output[0].cpu().numpy(), trunk_target[0].cpu().numpy())
                ncc = NCC(output[0], trunk_target[0]).cpu().numpy()
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(output[0].cpu().numpy().reshape(eval_config.dataset.image_size))
                plt.title(f'Model Output, NCC: {ncc:.2f}')

                plt.subplot(1,2,2)
                plt.imshow(trunk_target[0].cpu().numpy().reshape(eval_config.dataset.image_size))
                plt.title('Ground Truth')
                np.savetxt('proposed_64.txt', output[0].cpu().numpy().reshape(eval_config.dataset.image_size))
                

                plt.savefig('example.png')
                plt.close()
            batch_mse = nmse_tot(output.cpu().numpy(), trunk_target.cpu().numpy())
            batch_ncc = NCC(output, trunk_target).cpu().numpy()

            test_mse += batch_mse
            test_ncc += batch_ncc
            bn+=1
    test_mse /= len(test_dataloader)
    test_ncc /= len(test_dataloader)
    print(f"Test NMSE: {test_mse} dB")
    print(f"Test NCC: {test_ncc} dB")
    return test_mse, test_ncc

if __name__ == "__main__":
    load_dotenv()
    evaluate()