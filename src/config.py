"""This is the configuration file which loads the model's parameters from the file. 
"""
#!/usr/bin/env python
import argparse
from configparser import ConfigParser
from utils import convert_to_bool

class Config:
    """The configuration class
    """

    def __init__(self, paramters_filepath):
        """reads the parameters from the parameters filepath

        Args:
            paramters_filepath (filepath): the filepath of the parameters.ini for config

        Returns:
            dict: parameters for the file
        """

        parser = ConfigParser()
        parser.read_file(open(paramters_filepath))
        
        # reading the options for model - training and testing
        self.train_file = parser['model']['train_file']
        self.dev_file = parser['model']['dev_file']
        self.dev_key = parser['model']['dev_key']
        self.test_file = parser['model']['test_file']
        self.test_key = parser['model']['test_key']
        self.output_predictions = parser['model']['output_predictions']

        # hyperparamters and other options 
        self.use_dev = convert_to_bool(parser['options']['use_dev'])
        self.epochs = int(parser['options']['nepochs'])
        self.sigma = float(parser['options']['sigma'])
        self.learning_rate = float(parser['options']['learning_rate'])

        # context features; this indicates which features have to be used
        self.use_prev_current_token = convert_to_bool(parser['context_features']['use_prev_current_token'])
        self.use_current_next_token = convert_to_bool(parser['context_features']['use_current_next_token'])
        self.use_prev_current_metaphone = convert_to_bool(parser['context_features']['use_prev_current_metaphone'])
        self.use_current_next_metaphone = convert_to_bool(parser['context_features']['use_current_next_metaphone'])
        self.use_prev_current_pos = convert_to_bool(parser['context_features']['use_prev_current_pos'])
        self.use_current_next_pos = convert_to_bool(parser['context_features']['use_current_next_pos'])
        self.use_first_token = convert_to_bool(parser['context_features']['use_first_token'])
