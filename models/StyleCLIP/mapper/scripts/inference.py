import os
import pickle
from argparse import Namespace
import torchvision
import torch
import sys
import time

from configs import paths_config, global_config
from models.StyleCLIP.mapper.styleclip_mapper import StyleCLIPMapper
from utils.models_utils import load_tuned_G, load_old_G

sys.path.append(".")
sys.path.append("..")


def run(test_opts, model_id, image_name, use_multi_id_G):
    out_path_results = os.path.join(test_opts.exp_dir, test_opts.data_dir_name)
    os.makedirs(out_path_results, exist_ok=True)
    out_path_results = os.path.join(out_path_results, test_opts.image_name)
    os.makedirs(out_path_results, exist_ok=True)

    # update test configs with configs used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    net = StyleCLIPMapper(opts, test_opts.run_id)
    net.eval()
    net.to(global_config.device)

    generator_type = paths_config.multi_id_model_type if use_multi_id_G else image_name

    new_G = load_tuned_G(model_id, generator_type)
    old_G = load_old_G()

    run_styleclip(net, new_G, opts, paths_config.pti_results_keyword, out_path_results, test_opts)
    run_styleclip(net, old_G, opts, paths_config.e4e_results_keyword, out_path_results, test_opts)


def run_styleclip(net, G, opts, method, out_path_results, test_opts):
    net.set_G(G)

    out_path_results = os.path.join(out_path_results, method)
    os.makedirs(out_path_results, exist_ok=True)

    latent = torch.load(opts.latents_test_path)

    global_i = 0
    global_time = []
    with torch.no_grad():
        input_cuda = latent.cuda().float()
        tic = time.time()
        result_batch = run_on_batch(input_cuda, net, test_opts.couple_outputs)
        toc = time.time()
        global_time.append(toc - tic)

    for i in range(opts.test_batch_size):
        im_path = f'{test_opts.image_name}_{test_opts.edit_name}'
        if test_opts.couple_outputs:
            couple_output = torch.cat([result_batch[2][i].unsqueeze(0), result_batch[0][i].unsqueeze(0)])
            torchvision.utils.save_image(couple_output, os.path.join(out_path_results, f"{im_path}.jpg"),
                                         normalize=True, range=(-1, 1))
        else:
            torchvision.utils.save_image(result_batch[0][i], os.path.join(out_path_results, f"{im_path}.jpg"),
                                         normalize=True, range=(-1, 1))
        torch.save(result_batch[1][i].detach().cpu(), os.path.join(out_path_results, f"latent_{im_path}.pt"))


def run_on_batch(inputs, net, couple_outputs=False):
    w = inputs
    with torch.no_grad():
        w_hat = w + 0.06 * net.mapper(w)
        x_hat = net.decoder.synthesis(w_hat, noise_mode='const', force_fp32=True)
        result_batch = (x_hat, w_hat)
        if couple_outputs:
            x = net.decoder.synthesis(w, noise_mode='const', force_fp32=True)
            result_batch = (x_hat, w_hat, x)
    return result_batch
