import torch
from configs import global_config, paths_config
from scripts.latent_creators.base_latent_creator import BaseLatentCreator
from training.projectors import w_projector


class SG2LatentCreator(BaseLatentCreator):

    def __init__(self, use_wandb=False, projection_steps=600):
        super().__init__(paths_config.sg2_results_keyword, use_wandb=use_wandb)

        self.projection_steps = projection_steps

    def run_projection(self, fname, image):
        image = torch.squeeze((image.to(global_config.device) + 1) / 2) * 255
        w = w_projector.project(self.old_G, image, device=torch.device(global_config.device),
                                num_steps=self.projection_steps, w_name=fname, use_wandb=self.use_wandb)

        return w


if __name__ == '__main__':
    id_change_report = SG2LatentCreator()
    id_change_report.create_latents()
