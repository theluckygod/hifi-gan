from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator
import tqdm
import time

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = glob.glob(f"{a.input_mels_dir}/*.npy")

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        total_time = 0.0
        for file_path in tqdm.tqdm(filelist):
            x = np.load(file_path)
            x = torch.FloatTensor(x).to(device)
            if len(x.shape) < 3:  # for mel from vivos
                x = x.unsqueeze(0)
            
            start_time = time.time()

            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().detach().numpy().astype('int16')

            total_time += time.time() - start_time

            file_name = os.path.basename(file_path).split('.')[0]
            output_file = os.path.join(a.output_dir, file_name + '_generated_from_mel.wav')
            write(output_file, h.sampling_rate, audio)

    print("Elapsed time:", total_time)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='test_mel_files')
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

