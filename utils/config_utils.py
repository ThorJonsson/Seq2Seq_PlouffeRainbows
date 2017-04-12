import argparse
import yaml
import os
import sys


class Config:
    """ 
    Main class which parses the arguments for neural network training
    """

    def __init__(self, yamlpath):
		#self.yamlpath = os.path.dirname(os.path.abspath(__file__)) + '/configs/' # YAML config file path
        self.yamlpath = yamlpath
        self.args = None # Model/Dataset parameters

    @staticmethod
    def config_init_parser(args=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config-file', type=yaml.load, help='training input configuration file')

        # Global Options
        globalArgs = parser.add_argument_group('Global options')
        '''
        # TODO
        globalArgs.add_argument('-t', '--test',
                                nargs='?',
                                choices=[],
                                const=, default=None,
                                help='if present, launch the program try to predict next set of integer sequences from data/test/ with')
        globalArgs.add_argument('-cd', '--createDataset', action='store_true', help='if present, the program will only generate the dataset for the specified algorithm')
        globalArgs.add_argument('-pd', '--playDataset', type=int, nargs='?', const=10, default=None,  help='if set, the program will randomly play some samples')
        globalArgs.add_argument('-v', '--verbose', action='store_true', help='when testing, this will plot the outputs at the same time they are computed')
        globalArgs.add_argument('-mt, '--modelTag', type=str, default=None, help='tag to differentiate which model to store/load')
        globalArgs.add_argument('--rootDir', type=str, default=None, help='folder where to look for the models and data')
        '''
        #globalArgs.add_argument('--train', type=str, default='local', help='Training on sharcnet or local')
        globalArgs.add_argument('--checkpointDir', type=str, help='Set checkpoint directory')
        globalArgs.add_argument('--checkpointName', type=str, help='Set checkpoint name')

		
		# Dataset options
        datasetArgs = parser.add_argument_group('Dataset options')
        datasetArgs.add_argument('--datasetSize', type=int, default=1000, help='the size of the dataset being created')
        '''
        # TODO
        datasetArgs.add_argument('-sd', '--syntheticDataset', type=str, default='fibonacci', help='the algorithmic task to which to generate the synthetic dataset.')
        datasetArgs.add_argument('-dt', '--datasetTag', type=str, default=None, help='add a tag to the dataset. Useful to manage multiple versions')
        datasetArgs.add_argument('-rd', '--ratioDataset', type=float, default=1.0, help='ratio of dataset used to avoid using the whole dataset')
        datasetArgs.add_argument('-ml', '--maxLength', type=int, default=10, help='maximum length of the sentence (for input and output), define number of maximum step of the RNN')
		'''

		# Network Options 
        nnArgs = parser.add_argument_group('Network options', 'architecture related option')
        nnArgs.add_argument('--cellSize', type=int, default=64, help='size of RNN cell')
        '''
        # TODO
         nnArgs.add_argument('-hs', '--hiddenSize', type=int, default=256, help='number of hidden units in each RNN cell')
        nnArgs.add_argument('-nl', '--numLayers', type=int, default=2, help='number of rnn layers')
        '''

        # Training Options
        trainingArgs = parser.add_argument_group('Training options')
        trainingArgs.add_argument('--batchSize', type=int, default=100, help='mini-batch size')
        trainingArgs.add_argument('--maxEpoch', type=int, default=600, help='maximum number of epochs to run')
        trainingArgs.add_argument('--saveEvery', type=int, default=10, help='number of mini-batch step before creating a model checkpoint')
        trainingArgs.add_argument('--validateEvery', type=int, default=1, help='validate an epoch at every step')
        trainingArgs.add_argument('--learningRate', type=float, default=0.01, help='learning rate')
        trainingArgs.add_argument('--numNodes', type=int, default=100, help='number of nodes in Plouffe')
        trainingArgs.add_argument('--numFrames', type=int, default=200, help='number of frames in Plouffe')

        '''
        # TODO
        trainingArgs.add_argument('-lrd', '--learningRateDecay', type=int, default=1, help='learning rate decay')
		trainingArgs.add_argument('-im', '--initMomentum', type=float, default=0.5, help='initial momentum')
		trainingArgs.add_argument('-fm', '--finalMomentum', type=float, default=0.9, help='final momentum')
        '''
        return parser.parse_args(args)

    def config_parse_yaml(self):
    	'''
    	Parses the YAML training input config file
    	'''

    	# Parse command line arguments
    	args = None
    	self.args = self.config_init_parser(args)
    	
    	# Load YAML configuration file
    	with open(self.yamlpath + '/'+ self.args.config_file) as f:
    		ydict = yaml.safe_load(f)

    	# Add list attributes from args to the corresponding ydict values
		for k, v in ydict.items():
			av = getattr(args, k, None)
			if av and isinstance(v, list):
				v.extend(av)

		return (ydict)

def flat_dict_helper(prepend,d):
    if len(prepend) > 0:
        prepend = prepend + "."
    for k in d:
        i=d[k]
        if type(i).__name__=='dict':
            r = flat_dict_helper(prepend+k,i)
            for j in r:
                yield j
        else:
            yield (prepend+k,i)

def flat_dict(d): return dict(flat_dict_helper("",d))

if __name__ == "__main__":
	config = Config()
	sess_args = config.config_parse_yaml()
	print(flat_dict(sess_args))
