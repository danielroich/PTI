import os
from random import choice
from string import ascii_uppercase
from PIL import Image
from tqdm import tqdm
from scripts.latent_editor_wrapper import LatentEditorWrapper
from evaluation.experiment_setting_creator import ExperimentRunner
import torch
from configs import paths_config, hyperparameters, evaluation_config
from utils.log_utils import save_concat_image, save_single_image
from utils.models_utils import load_tuned_G


class EditComparison:

    def __init__(self, save_single_images, save_concatenated_images, run_id):

        self.run_id = run_id
        self.experiment_creator = ExperimentRunner(run_id)
        self.save_single_images = save_single_images
        self.save_concatenated_images = save_concatenated_images
        self.latent_editor = LatentEditorWrapper()

    def save_reconstruction_images(self, image_latents, new_inv_image_latent, new_G, target_image):
        if self.save_concatenated_images:
            save_concat_image(self.concat_base_dir, image_latents, new_inv_image_latent, new_G,
                              self.experiment_creator.old_G,
                              'rec',
                              target_image)

        if self.save_single_images:
            save_single_image(self.single_base_dir, new_inv_image_latent, new_G, 'rec')
            target_image.save(f'{self.single_base_dir}/Original.jpg')

    def create_output_dirs(self, full_image_name):
        output_base_dir_path = f'{paths_config.experiments_output_dir}/{paths_config.input_data_id}/{self.run_id}/{full_image_name}'
        os.makedirs(output_base_dir_path, exist_ok=True)

        self.concat_base_dir = f'{output_base_dir_path}/concat_images'
        self.single_base_dir = f'{output_base_dir_path}/single_images'

        os.makedirs(self.concat_base_dir, exist_ok=True)
        os.makedirs(self.single_base_dir, exist_ok=True)

    def get_image_latent_codes(self, image_name):
        image_latents = []
        for method in evaluation_config.evaluated_methods:
            if method == 'SG2':
                image_latents.append(torch.load(
                    f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/'
                    f'{paths_config.pti_results_keyword}/{image_name}/0.pt'))
            else:
                image_latents.append(torch.load(
                    f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{method}/{image_name}/0.pt'))
        new_inv_image_latent = torch.load(
            f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}/{image_name}/0.pt')

        return image_latents, new_inv_image_latent

    def save_interfacegan_edits(self, image_latents, new_inv_image_latent, interfacegan_factors, new_G, target_image):
        new_w_inv_edits = self.latent_editor.get_single_interface_gan_edits(new_inv_image_latent,
                                                                            interfacegan_factors)

        inv_edits = []
        for latent in image_latents:
            inv_edits.append(self.latent_editor.get_single_interface_gan_edits(latent, interfacegan_factors))

        for direction, edits in new_w_inv_edits.items():
            for factor, edit_tensor in edits.items():
                if self.save_concatenated_images:
                    save_concat_image(self.concat_base_dir, [edits[direction][factor] for edits in inv_edits],
                                      new_w_inv_edits[direction][factor],
                                      new_G,
                                      self.experiment_creator.old_G,
                                      f'{direction}_{factor}', target_image)
                if self.save_single_images:
                    save_single_image(self.single_base_dir, new_w_inv_edits[direction][factor], new_G,
                                      f'{direction}_{factor}')

    def save_ganspace_edits(self, image_latents, new_inv_image_latent, factors, new_G, target_image):
        new_w_inv_edits = self.latent_editor.get_single_ganspace_edits(new_inv_image_latent, factors)
        inv_edits = []
        for latent in image_latents:
            inv_edits.append(self.latent_editor.get_single_ganspace_edits(latent, factors))

        for idx in range(len(new_w_inv_edits)):
            if self.save_concatenated_images:
                save_concat_image(self.concat_base_dir, [edit[idx] for edit in inv_edits], new_w_inv_edits[idx],
                                  new_G,
                                  self.experiment_creator.old_G,
                                  f'ganspace_{idx}', target_image)
            if self.save_single_images:
                save_single_image(self.single_base_dir, new_w_inv_edits[idx], new_G,
                                  f'ganspace_{idx}')

    def run_experiment(self, run_pt, create_other_latents, use_multi_id_training, use_wandb=False):
        images_counter = 0
        new_G = None
        interfacegan_factors = [val / 2 for val in range(-6, 7) if val != 0]
        ganspace_factors = range(-20, 25, 5)
        self.experiment_creator.run_experiment(run_pt, create_other_latents, use_multi_id_training, use_wandb)

        if use_multi_id_training:
            new_G = load_tuned_G(self.run_id, paths_config.multi_id_model_type)

        for idx, image_path in tqdm(enumerate(self.experiment_creator.images_paths),
                                    total=len(self.experiment_creator.images_paths)):

            if images_counter >= hyperparameters.max_images_to_invert:
                break

            image_name = image_path.split('.')[0].split('/')[-1]
            target_image = Image.open(self.experiment_creator.target_paths[idx])

            if not use_multi_id_training:
                new_G = load_tuned_G(self.run_id, image_name)

            image_latents, new_inv_image_latent = self.get_image_latent_codes(image_name)

            self.create_output_dirs(image_name)

            self.save_reconstruction_images(image_latents, new_inv_image_latent, new_G, target_image)

            self.save_interfacegan_edits(image_latents, new_inv_image_latent, interfacegan_factors, new_G, target_image)

            self.save_ganspace_edits(image_latents, new_inv_image_latent, ganspace_factors, new_G, target_image)

            target_image.close()
            torch.cuda.empty_cache()
            images_counter += 1


def run_pti_and_full_edit(iid):
    evaluation_config.evaluated_methods = ['SG2Plus', 'e4e', 'SG2']
    edit_figure_creator = EditComparison(save_single_images=True, save_concatenated_images=True,
                                         run_id=f'{paths_config.input_data_id}_pti_full_edit_{iid}')
    edit_figure_creator.run_experiment(True, True, use_multi_id_training=False, use_wandb=False)


def pti_no_comparison(iid):
    evaluation_config.evaluated_methods = []
    edit_figure_creator = EditComparison(save_single_images=True, save_concatenated_images=True,
                                         run_id=f'{paths_config.input_data_id}_pti_no_comparison_{iid}')
    edit_figure_creator.run_experiment(True, False, use_multi_id_training=False, use_wandb=False)


def edits_for_existed_experiment(run_id):
    evaluation_config.evaluated_methods = ['SG2Plus', 'e4e', 'SG2']
    edit_figure_creator = EditComparison(save_single_images=True, save_concatenated_images=True,
                                         run_id=run_id)
    edit_figure_creator.run_experiment(False, True, use_multi_id_training=False, use_wandb=False)


if __name__ == '__main__':
    iid = ''.join(choice(ascii_uppercase) for i in range(7))
    pti_no_comparison(iid)
