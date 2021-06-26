import glob
from argparse import Namespace
from configs import paths_config
from models.StyleCLIP.mapper.scripts.inference import run
from scripts.run_pti import run_PTI

meta_data = {
    'afro': ['afro', False, False, True],
    'angry': ['angry', False, False, True],
    'Beyonce': ['beyonce', False, False, False],
    'bobcut': ['bobcut', False, False, True],
    'bowlcut': ['bowlcut', False, False, True],
    'curly hair': ['curly_hair', False, False, True],
    'Hilary Clinton': ['hilary_clinton', False, False, False],
    'Jhonny Depp': ['depp', False, False, False],
    'mohawk': ['mohawk', False, False, True],
    'purple hair': ['purple_hair', False, False, False],
    'surprised': ['surprised', False, False, True],
    'Taylor Swift': ['taylor_swift', False, False, False],
    'trump': ['trump', False, False, False],
    'Mark Zuckerberg': ['zuckerberg', False, False, False]
}


def styleclip_edit(use_multi_id_G, run_id, use_wandb, edit_types):
    images_dir = paths_config.input_data_path
    pretrained_mappers = paths_config.style_clip_pretrained_mappers
    data_dir_name = paths_config.input_data_id
    if run_id == '':
        run_id = run_PTI(run_name='', use_wandb=use_wandb, use_multi_id_training=False)
    images = glob.glob(f"{images_dir}/*.jpeg")
    w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
    for image_name in images:
        image_name = image_name.split(".")[0].split("/")[-1]
        embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        latent_path = f'{embedding_dir}/0.pt'
        for edit_type in set(meta_data.keys()).intersection(edit_types):
            edit_id = meta_data[edit_type][0]
            args = {
                "exp_dir": f'{paths_config.styleclip_output_dir}',
                "checkpoint_path": f"{pretrained_mappers}/{edit_id}.pt",
                "couple_outputs": False,
                "mapper_type": "LevelsMapper",
                "no_coarse_mapper": meta_data[edit_type][1],
                "no_medium_mapper": meta_data[edit_type][2],
                "no_fine_mapper": meta_data[edit_type][3],
                "stylegan_size": 1024,
                "test_batch_size": 1,
                "latents_test_path": latent_path,
                "test_workers": 1,
                "run_id": run_id,
                "image_name": image_name,
                'edit_name': edit_type,
                "data_dir_name": data_dir_name
            }

            run(Namespace(**args), run_id, image_name, use_multi_id_G)
