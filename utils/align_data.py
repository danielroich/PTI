from configs import paths_config
import dlib
import glob
import os
from tqdm import tqdm
from utils.alignment import align_face


def pre_process_images(raw_images_path):
    current_directory = os.getcwd()

    IMAGE_SIZE = 1024
    predictor = dlib.shape_predictor(paths_config.dlib)
    os.chdir(raw_images_path)
    images_names = glob.glob(f'*')

    aligned_images = []
    for image_name in tqdm(images_names):
        try:
            aligned_image = align_face(filepath=f'{raw_images_path}/{image_name}',
                                       predictor=predictor, output_size=IMAGE_SIZE)
            aligned_images.append(aligned_image)
        except Exception as e:
            print(e)

    os.makedirs(paths_config.input_data_path, exist_ok=True)
    for image, name in zip(aligned_images, images_names):
        real_name = name.split('.')[0]
        image.save(f'{paths_config.input_data_path}/{real_name}.jpeg')

    os.chdir(current_directory)


if __name__ == "__main__":
    pre_process_images('')
