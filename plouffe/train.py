import sys
sys.path.append("../")

from seq2seq_regression.Seq2SeqRegression import train_on_plouffe_copy
from utils.config_utils import Config, flat_dict, flat_dict_helper

if __name__=="__main__":
    #train_on_plouffe_copy()
    config = Config()
    sess_args = config.config_parse_yaml()
    print(flat_dict(sess_args))
