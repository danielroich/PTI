import torch
from configs import paths_config
from editings.latent_editor import LatentEditor


class LatentEditorWrapper:

    def __init__(self):

        self.interfacegan_directions = {'age': f'{paths_config.interfacegan_age}',
                                        'smile': f'{paths_config.interfacegan_smile}',
                                        'rotation': f'{paths_config.interfacegan_rotation}'}
        self.interfacegan_directions_tensors = {name: torch.load(path).cuda() for name, path in
                                                self.interfacegan_directions.items()}
        self.ganspace_pca = torch.load(f'{paths_config.ffhq_pca}')

        ## For more edit directions please visit ..
        self.ganspace_directions = {
            'eye_openness': (54, 7, 8, 5),
            'smile': (46, 4, 5, -6),
            'trimmed_beard': (58, 7, 9, 7),
        }

        self.latent_editor = LatentEditor()

    def get_single_ganspace_edits(self, start_w, factors):
        latents_to_display = []
        for ganspace_direction in self.ganspace_directions.values():
            for factor in factors:
                edit_direction = list(ganspace_direction)
                edit_direction[-1] = factor
                edit_direction = tuple(edit_direction)
                new_w = self.latent_editor.apply_ganspace(start_w, self.ganspace_pca, [edit_direction])
                latents_to_display.append(new_w)
        return latents_to_display

    def get_single_interface_gan_edits(self, start_w, factors):
        latents_to_display = {}
        for direction in ['rotation', 'smile', 'age']:
            for factor in factors:
                if direction not in latents_to_display:
                    latents_to_display[direction] = {}
                latents_to_display[direction][factor] = self.latent_editor.apply_interfacegan(
                    start_w, self.interfacegan_directions_tensors[direction], factor / 2)

        return latents_to_display
