import os
import sys
sys.path.append("../")

from seq2seq_regression.Seq2SeqRegression import train_on_plouffe_copy
from utils.config_utils import Config, flat_dict, flat_dict_helper
from job_runner import train_many_jobs

if __name__=="__main__":
    config_path = os.getcwd()
    config = Config(config_path)
    yml_args = config.config_parse_yaml()
    sess_args = flat_dict(yml_args)

    train_many_jobs(sess_args)
