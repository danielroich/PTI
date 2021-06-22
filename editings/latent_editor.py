import torch

from configs import paths_config
from editings import ganspace
from utils.data_utils import tensor2im


class LatentEditor(object):

    def apply_ganspace(self, latent, ganspace_pca, edit_directions):
        edit_latents = ganspace.edit(latent, ganspace_pca, edit_directions)
        return edit_latents

    def apply_interfacegan(self, latent, direction, factor=1, factor_range=None):
        edit_latents = []
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latent + f * direction
                edit_latents.append(edit_latent)
            edit_latents = torch.cat(edit_latents)
        else:
            edit_latents = latent + factor * direction
        return edit_latents
