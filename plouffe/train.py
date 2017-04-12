import os
import sys
sys.path.append("../")

from seq2seq_regression.Seq2SeqRegression import train_on_plouffe_copy
from utils.config_utils import Config, flat_dict, flat_dict_helper

if __name__=="__main__":
    config_path = os.getcwd()
    config = Config(config_path)

    cmd_args = config.config_init_parser()
    load_params = 0

    if cmd_args.config_file is None:
        sess_args = vars(cmd_args)
        load_params = 1
    else:
        yml_args = config.config_parse_yaml()
        sess_args = flat_dict(yml_args)

    train_on_plouffe_copy(sess_args, load_params)
