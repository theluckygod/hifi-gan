import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
import warnings
warnings.simplefilter("ignore", UserWarning)

torch.backends.cudnn.benchmark = True


def train(rank, args, cfg):
    if cfg.num_gpus > 1:
        init_process_group(backend=cfg.dist_config['dist_backend'], init_method=cfg.dist_config['dist_url'],
                           world_size=cfg.dist_config['world_size'] * cfg.num_gpus, rank=rank)

    torch.cuda.manual_seed(cfg.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(cfg).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(args.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", args.checkpoint_path)

    if os.path.isdir(args.checkpoint_path):
        cp_g = scan_checkpoint(args.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(args.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if cfg.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), cfg.learning_rate, betas=[cfg.adam_b1, cfg.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                cfg.learning_rate, betas=[cfg.adam_b1, cfg.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=cfg.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=cfg.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(args)

    trainset = MelDataset(training_filelist, cfg.segment_size, cfg.n_fft, cfg.num_mels,
                          cfg.hop_size, cfg.win_size, cfg.sampling_rate, cfg.fmin, cfg.fmax, n_cache_reuse=0,
                          shuffle=False if cfg.num_gpus > 1 else True, fmax_loss=cfg.fmax_for_loss, device=device,
                          fine_tuning=args.fine_tuning, base_mels_path=args.train_mels_dir, textgrid_path=args.textgrid_path)

    train_sampler = DistributedSampler(trainset) if cfg.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=cfg.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=cfg.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, cfg.segment_size, cfg.n_fft, cfg.num_mels,
                              cfg.hop_size, cfg.win_size, cfg.sampling_rate, cfg.fmin, cfg.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=cfg.fmax_for_loss, device=device, fine_tuning=args.fine_tuning,
                              base_mels_path=args.val_mels_dir, textgrid_path=args.textgrid_path)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(args.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), args.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if cfg.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, y_mel, _ = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), cfg.n_fft, cfg.num_mels, cfg.sampling_rate, cfg.hop_size, cfg.win_size,
                                          cfg.fmin, cfg.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % args.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
                if steps % args.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(args.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if cfg.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(args.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'mpd': (mpd.module if cfg.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if cfg.num_gpus > 1
                                                         else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % args.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % args.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, y_mel, _ = batch
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), cfg.n_fft, cfg.num_mels, cfg.sampling_rate,
                                                          cfg.hop_size, cfg.win_size,
                                                          cfg.fmin, cfg.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, cfg.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, cfg.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), cfg.n_fft, cfg.num_mels,
                                                             cfg.sampling_rate, cfg.hop_size, cfg.win_size,
                                                             cfg.fmin, cfg.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--train_wavs_dir', default='vlsp_2021/wav')
    parser.add_argument('--val_wavs_dir', default='vlsp_2021/wav')
    parser.add_argument('--train_mels_dir', default='vlsp_2021/mel')
    parser.add_argument('--val_mels_dir', default='vlsp_2021/mel')
    parser.add_argument('--input_training_file', default='vlsp_2021/train.txt')
    parser.add_argument('--input_validation_file', default='vlsp_2021/val.txt')
    parser.add_argument('--textgrid_path', help="Only for fine tune FastSpeech2 mel", default='vlsp_2021/TextGrid')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='config_v1_vlsp_2021.json')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=10, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default="", type=str)

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    json_config = json.loads(data)
    cfg = AttrDict(json_config)
    build_env(args.config, 'config.json', args.checkpoint_path)

    torch.cuda.empty_cache()

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        cfg.num_gpus = torch.cuda.device_count()
        cfg.batch_size = int(cfg.batch_size / cfg.num_gpus)
        print('Batch size per GPU :', cfg.batch_size)
    else:
        pass

    if cfg.num_gpus > 1:
        mp.spawn(train, nprocs=cfg.num_gpus, args=(args, cfg,))
    else:
        train(0, args, cfg)


if __name__ == '__main__':
    main()
