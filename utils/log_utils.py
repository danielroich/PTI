from PIL import Image
import wandb
from configs import global_config
import torch
import matplotlib.pyplot as plt


def log_image_from_w(w, G, name):
    img = get_image_from_w(w, G)
    pillow_image = Image.fromarray(img)
    wandb.log(
        {f"{name}": [
            wandb.Image(pillow_image, caption=f"current inversion {name}")]},
        step=global_config.training_step)


def log_images_from_w(ws, G, names):
    for name, w in zip(names, ws):
        w = w.to(global_config.device)
        log_image_from_w(w, G, name)


def plot_image_from_w(w, G):
    img = get_image_from_w(w, G)
    pillow_image = Image.fromarray(img)
    plt.imshow(pillow_image)
    plt.show()


def plot_image(img):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    pillow_image = Image.fromarray(img[0])
    plt.imshow(pillow_image)
    plt.show()


def get_image_from_w(w, G):
    if len(w.size()) <= 2:
        w = w.unsqueeze(0)
    with torch.no_grad():
        img = G.synthesis(w, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    return img[0]
