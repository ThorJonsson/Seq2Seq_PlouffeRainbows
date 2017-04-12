#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import json
import random
import numpy as np
import pdb
"""
List of Icelandic Volcanoes.
"""
volcanoes = ['Gunnuhver',
             'Trölladyngja',
             'Hengill',
             'Hrómundartindur',
             'Seyðishólar',
             'Laugarfjall',
             'Prestahnúkur',
             'Hveravellir',
             'Hofsjökull',
             'Snækollur',
             'Tungnafellsjökull',
             'Eyjafjallajökull',
             'Katla',
             'Tindfjallajökull',
             'Hekla',
             'Torfajökull',
             'Bárðarbunga',
             'Thórdarhyrna',
             'Vonarskard',
             'Kverkfjöll',
             'Askja',
             'Krafla',
             'Þeistareykjabunga',
             'Öræfajökull',
             'Snæhetta',
             'Snæfell',
             'Helgrindur',
             'Snæfellsjökull']


def train_from_config(learning_rate,
                      batch_size,
                      radius,
                      checkpoint_name,
                      log_dir_num,
                      argv=None):
    """Runs `train.py` either on copper or locally using a set of
    parameters taken from an input JSON config file.

    i.e. running on copper
        python train_from_config.py config.json copper

    i.e. running locally
        python train_from_config.py config.json local

    i.e. running manually
        python train.py
    """

    assert len(argv) == 3, "Usage: python train.py [copper_config.json] [copper/local]"

    # Get basic stuff from JSON file
    with open(argv[1]) as config_file:
        config = json.load(config_file)

    ##########
    # Make config from json + hyperparameter search
    ##########
    config_string = ""
    # reset log_dir s.t. we have one dir for each job
    config['log_dir'] += '/' + str(log_dir_num)
    for option in config:
        config_string += ' --' + option + ' ' + str(config[option])
    ##########
    # Set hyperparameters
    ##########
    config_string += ' --' + 'initial_learning_rate' + ' ' + str(learning_rate)
    config_string += ' --' + 'batch_size' + ' ' + str(batch_size)
    config_string += ' --' + 'heatmap_stddev_pixels' + ' ' + str(radius)
    config_string += ' --' + 'checkpoint_name' + ' ' + checkpoint_name

    print(config_string)
    command = 'python3 -m train' + config_string
    ##########
    # Check if log directory exists
    ##########
    log_dir = config['log_dir']
    if os.path.exists(log_dir):
        print('Using logging directory ' + log_dir)
    else:
        print('Logging directory doesnt exist, creating ' + log_dir)
        os.mkdir(log_dir)

    # For jobscheduler
    if argv[2] == 'copper':
        sqsub = 'sqsub -q gpu -f mpi -n 8 --gpp 4 -r 3600 -o ' + log_dir
        sqsub += '/' + checkpoint_name + '%J.out --mpp=92g --nompirun '
        #print(sqsub + command)
        subprocess.call(sqsub + command, shell=True)

    elif argv[2] == 'local':
        #print(command)
        subprocess.call(command, shell=True)

#TODO shuffle name list
def train_many_jobs(name_list):
    # Number of log directory
    log_dir_num = 0
    lr_bin = [10**(-3), 10**(-4)]
    batch_size_bin = [32, 48, 64, 80]
    heatmap_radius_bin = [5, 10, 15]
    # Random hyperparameter search for the learning rate, the batch size and
    # the heatmap radius
    for i,lr in enumerate(lr_bin):
        # For random search
        # learning_rate = np.random.uniform(lr, lr*10)
        for j,batch_size in enumerate(batch_size_bin):
            # We assume we're using 4 gpus
            # For random search
            # batch_size = 4*np.random.randint(mb_size, mb_size*2)
            for k, heatmap_radius in enumerate(heatmap_radius_bin):
                # for random search
                # radius = np.random.randint(heatmap_radius, heatmap_radius*2)
                log_dir_num += 1
                print('Starting experiment', log_dir_num)
                print('# Hyperparameters:')
                print('initial learning_rate:', lr)
                print('batch size:', batch_size)
                print('radius:', heatmap_radius)
                checkpoint_name = name_list[log_dir_num]+str(np.random.randint(0,1000))
                train_from_config(lr,
                                  batch_size,
                                  heatmap_radius,
                                  checkpoint_name,
                                  log_dir_num,
                                  sys.argv)

if __name__ == "__main__":
    train_many_jobs(volcanoes)

