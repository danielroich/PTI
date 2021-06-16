import torch
from argparse import Namespace
from torchvision.transforms import transforms

from configs import paths_config
from models.e4e.psp import pSp
from scripts.latent_creators.base_latent_creator import BaseLatentCreator
from utils.log_utils import log_image_from_w


class E4ELatentCreator(BaseLatentCreator):

    def __init__(self, use_wandb=False):
        self.e4e_inversion_pre_process = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        super().__init__('e4e', self.e4e_inversion_pre_process, use_wandb=use_wandb)

        e4e_model_path = paths_config.e4e
        ckpt = torch.load(e4e_model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['batch_size'] = 1
        opts['checkpoint_path'] = e4e_model_path
        opts = Namespace(**opts)
        self.e4e_inversion_net = pSp(opts)
        self.e4e_inversion_net.eval()
        self.e4e_inversion_net = self.e4e_inversion_net.cuda()

    def run_projection(self, fname, image):
        _, e4e_image_latent = self.e4e_inversion_net(image, randomize_noise=False, return_latents=True,
                                                     resize=False,
                                                     input_code=False)

        if self.use_wandb:
            log_image_from_w(e4e_image_latent, self.old_G, 'First e4e inversion')

        return e4e_image_latent


if __name__ == '__main__':
    e4e_latent_creator = E4ELatentCreator()
    e4e_latent_creator.create_latents()
