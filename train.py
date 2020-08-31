"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import time, torch, os, json
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import numpy as np
from matplotlib import pyplot as plt
import torchvision
from tqdm import tqdm
import pandas as pd
from torch.utils import tensorboard
import seaborn as sn
from util.util import Unnormalize

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    if opt.seed:
        print('Using specified random seed:', opt.seed)
        torch.manual_seed(opt.seed)

    # ---- load datasets ----------------
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # ---------- validation dataset ------
    if opt.validation:
        opt.isTrain = False # to send the dataset validation signal
        opt.serial_batches = True
        opt.dataset_mode = opt.validation_datamode
        opt.collate = opt.validation_collate
        dataset_valid = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        print('The number of validation images = %d' % dataset_valid.__len__())
        opt.isTrain = True
    # -----------------------------------

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.train()

    total_iters = 0                # the total number of training iterations
    if opt.continue_train:
        total_iters = opt.epoch_count * dataset.__len__()

    print('Starting training (1st epoch)...')

    for epoch in range(opt.epoch_count, opt.max_epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        for i, data in enumerate(dataset):  # inner loop within one epoch

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # ----------- train -------------------------------------
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            # -------------------------------------------------------

            # ------ save models and print losses -------------------
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                print(f'Epoch {epoch}, iter {epoch_iter}, batch scores {tuple(losses.keys())}: {tuple(losses.values())}')
            # -------------------------------------------------------

            iter_data_time = time.time()

        # ----------------- end of an epoch -----------------------------
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.max_epochs, time.time() - epoch_start_time))

        # ------------------ update lr ---------------------
        latest_lr = model.update_learning_rate()                     # update learning rates at the end of every epoch.

        # -------------------------------------------------------
        # -------------------------------------------------------
        # ----------- validation --------------------------------
        # -------------------------------------------------------
        # -------------------------------------------------------

        if opt.validation: # at the end of every epoch
            losses = model.get_current_losses() # just to create a loss log dict with the keys!
            valid_log = {k:[] for k in losses.keys()}

            model.eval() # put model on eval mode
            
            for i, data in enumerate(dataset_valid):
                model.set_input(data)
                model.test()
                for k, v in model.get_current_losses().items():
                    valid_log[k].append(v)

            valid_log = {'val_' + k:np.mean(v) for k, v in valid_log.items()}
            
            print(f'>>> Validation scores {tuple(valid_log.keys())}: {tuple(valid_log.values())}')

            model.train() # put model back on train mode
        # -------------------------------------------------------
        
