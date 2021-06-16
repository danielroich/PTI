import abc
import logging

import pickle

import os
from random import choice
from string import ascii_uppercase
import torch
from torch.utils.data import DataLoader
import wandb
from configs import global_config, paths_config
from tqdm import tqdm

from torchvision import transforms

from utils.ImagesDataset import ImagesDataset


class BaseLatentCreator:

    def __init__(self, method_name, dara_preprocess=None, use_wandb=False):
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
        self.use_wandb = use_wandb
        if use_wandb:
            run = wandb.init(project="personalized_stylegan", reinit=True, name=global_config.run_name)

        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

        if dara_preprocess is None:
            self.projection_preprocess = transforms.Compose([
                transforms.Resize(1024),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        else:
            self.projection_preprocess = dara_preprocess

        image_dataset = ImagesDataset(f'{paths_config.input_data_path}', self.projection_preprocess)
        self.image_dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False)

        base_latent_folder_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(base_latent_folder_path, exist_ok=True)
        self.latent_folder_path = f'{base_latent_folder_path}/{method_name}'
        os.makedirs(self.latent_folder_path, exist_ok=True)

        with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
            self.old_G = pickle.load(f)['G_ema'].cuda()

    @abc.abstractmethod
    def run_projection(self, fname, image):
        return None

    def create_latents(self):
        for fname, image in tqdm(self.image_dataloader):
            fname = fname[0]
            cur_latent_folder_path = f'{self.latent_folder_path}/{fname}'
            image = image.cuda()
            w = self.run_projection(fname, image)

            os.makedirs(cur_latent_folder_path, exist_ok=True)
            torch.save(w, f'{cur_latent_folder_path}/0.pt')
