import glob
import os
from configs import global_config, paths_config, hyperparameters
from scripts.latent_creators.sg2_plus_latent_creator import SG2PlusLatentCreator
from scripts.latent_creators.e4e_latent_creator import E4ELatentCreator
from scripts.run_pti import run_PTI
import pickle
import torch
from utils.models_utils import toogle_grad, load_old_G


class ExperimentRunner:

    def __init__(self, run_id=''):
        self.images_paths = glob.glob(f'{paths_config.input_data_path}/*')
        self.target_paths = glob.glob(f'{paths_config.input_data_path}/*')
        self.run_id = run_id
        self.sampled_ws = None

        self.old_G = load_old_G()

        toogle_grad(self.old_G, False)

    def run_experiment(self, run_pt, create_other_latents, use_multi_id_training, use_wandb=False):
        if run_pt:
            self.run_id = run_PTI(self.run_id, use_wandb=use_wandb, use_multi_id_training=use_multi_id_training)
        if create_other_latents:
            sg2_plus_latent_creator = SG2PlusLatentCreator(use_wandb=use_wandb)
            sg2_plus_latent_creator.create_latents()
            e4e_latent_creator = E4ELatentCreator(use_wandb=use_wandb)
            e4e_latent_creator.create_latents()

        torch.cuda.empty_cache()

        return self.run_id


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    runner = ExperimentRunner()
    runner.run_experiment(True, False, False)
