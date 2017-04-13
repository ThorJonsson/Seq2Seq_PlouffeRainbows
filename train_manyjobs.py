#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import json
import random
import numpy as np
import pdb

from seq2seq_regression.Seq2SeqRegression import train_on_plouffe_copy
from utils.config_utils import Config, flat_dict, flat_dict_helper

def train_from_config(learning_rate,
                      batch_size,
                      num_nodes,
                      checkpoint_name,
                      log_dir_num,
                      log_dir_path,
                      train_option,
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

    config_string = ""

    # reset log_dir s.t. we have one dir for each job
    log_dir_path += '/exp' + str(log_dir_num)
    #print (log_dir_path)

    ##########
    # Set hyperparameters
    ##########
    config_string += ' --' + 'learningRate' + ' ' + str(learning_rate)
    config_string += ' --' + 'batchSize' + ' ' + str(batch_size)
    config_string += ' --' + 'numNodes' + ' ' + str(num_nodes)
    config_string += ' --' + 'checkpointName' + ' ' + str(checkpoint_name)
    config_string += ' --' + 'checkpointDir' + ' ' + str(log_dir_path)

    print(config_string)
    #print(log_dir_name)

    command = 'python -m train' + config_string
    #print(command)

    ##########
    # Check if log directory exists
    ##########
    if os.path.exists(log_dir_path):
        print('Using logging directory ' + log_dir_path)
    else:
        print('Logging directory doesnt exist, creating ' + log_dir_path)
        os.mkdir(log_dir_path)

    # For jobscheduler
    if train_option == 'copper':
        sqsub = 'sqsub -q gpu -f mpi -n 8 --gpp 1 -r 3600 -o ' + log_dir_path
        sqsub += '/' + checkpoint_name + '%J.out --mpp=92g --nompirun '
        print(sqsub + command)
        #exit()
        subprocess.call(sqsub + command, shell=True)

    elif train_option == 'local':
        #print(command)
        subprocess.call(command, shell=True)

#TODO shuffle name list
def train_many_jobs(sess_args):

    """
    List of Icelandic Volcanoes.
    """
    name_list = ['Gunnuhver',
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

    # Number of log directory
    log_dir_num = 1
    log_dir_path = '/work/thor/DLFractalSequences' + sess_args['globalParams.checkpointDir']
    #log_dir_path = os.getcwd() + sess_args['globalParams.checkpointDir']

    ##########
    # Check if log directory exists
    ##########
    if os.path.exists(log_dir_path):
        print('Using logging directory ' + log_dir_path)
    else:
        print('Logging directory doesnt exist, creating ' + log_dir_path)
        os.mkdir(log_dir_path)

    # Hyperparameters
    lr_bin = sess_args['hyperparameters.learningRate']
    num_frames_bin = sess_args['hyperparameters.numFrames']
    num_nodes_bin = sess_args['hyperparameters.numNodes']
    batch_size_bin = sess_args['hyperparameters.batchSize']
    train_option = sess_args['globalParams.train']

    # Random hyperparameter search for the learning rate, the batch size and
    # the heatmap radius
    for i, lr in enumerate(lr_bin):
        # For random search
        # learning_rate = np.random.uniform(lr, lr*10)
        for j, batch_size in enumerate(batch_size_bin):
            # We assume we're using 4 gpus
            # For random search
            # batch_size = 4*np.random.randint(mb_size, mb_size*2)
            for k, num_nodes in enumerate(num_nodes_bin):
                # for random search
                # radius = np.random.randint(heatmap_radius, heatmap_radius*2)
                print('Starting experiment', log_dir_num)
                print('# Hyperparameters:')
                print('learning_rate:', lr)
                print('batch size:', batch_size)
                print('num_nodes:', num_nodes)
                checkpoint_name = name_list[log_dir_num]+str(np.random.randint(0,1000))
                print(checkpoint_name)
                train_from_config(lr,
                                  batch_size,
                                  num_nodes,
                                  checkpoint_name,
                                  log_dir_num,
                                  log_dir_path,
                                  train_option,
                                  sys.argv)
                log_dir_num += 1

if __name__=="__main__":
    config_path = os.getcwd()
    config = Config(config_path)
    yml_args = config.config_parse_yaml()
    sess_args = flat_dict(yml_args)

    train_many_jobs(sess_args)
