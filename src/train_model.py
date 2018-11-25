"""
Duolingo SLAM Shared Task - Baseline Model

This code is written to be compatible with both Python 2 or 3, at the expense of dependency on the future library. This
code does not depend on any other Python libraries besides future.
"""

import argparse
from collections import defaultdict, namedtuple
from io import open
import math
import os
from random import shuffle, uniform

from future.builtins import range
from future.utils import iteritems
from metaphone import doublemetaphone
from prettytable import PrettyTable

from config import Config
from utils import load_context_json, load_labels

# Sigma is the L2 prior variance, regularizing the baseline model. Smaller sigma means more regularization.
_DEFAULT_SIGMA = 20.0

# Eta is the learning rate/step size for SGD. Larger means larger step size.
_DEFAULT_ETA = 0.1


def main():
    """
    This loads the training, dev(optional) and test data and returns the predictions
    """

    parser = argparse.ArgumentParser(description='Context based approach for second language acquisition')
    parser.add_argument('--params_file', required=True, help='The parameters file with all the options')
    args = parser.parse_args()

    config = Config(args.params_file)

    # load the context json - you will 
    context_train = load_context_json(config.train_file)
    context_dev = load_context_json(config.dev_file)
    context_test = load_context_json(config.test_file)

    training_data, training_labels = load_data(config.train_file, context_train)
    dev_data = load_data(config.dev_file, context_dev)
    dev_labels = load_labels(config.dev_key)
    test_data = load_data(config.test_file, context_test)

    # loading the train sets
    training_instances = [LogisticRegressionInstance(features=instance_data.to_features(config),
                                                     label=training_labels[instance_data.instance_id],
                                                     name=instance_data.instance_id
                                                     ) for instance_data in training_data]
    if config.use_dev:
        dev_instances = [LogisticRegressionInstance(features=instance_data.to_features(config),
                                                    label=dev_labels[instance_data.instance_id],
                                                    name=instance_data.instance_id
                                                    ) for instance_data in dev_data]
        
        training_instances += dev_instances

    # Loading the test set
    test_instances = [LogisticRegressionInstance(features=instance_data.to_features(config),
                                                 label=None,
                                                 name=instance_data.instance_id
                                                 ) for instance_data in test_data]

    # TODO: log all the details
    print("\n")
    print("Training Details: Context Features")

    # Using pretty table
    pt = PrettyTable()
    pt.field_names = ['Training Feature', 'True/False']
    pt.add_row(["Previous-Current Token POS (PCPOS)", config.use_prev_current_pos])
    pt.add_row(["Previous-Current Token Token (PCT)", config.use_prev_current_token])
    pt.add_row(["Previous-Current Token Metaphone (PCM)", config.use_prev_current_metaphone])

    pt.add_row(["Current-Next Token POS (CNPOS)", config.use_current_next_pos])
    pt.add_row(["Current-Next Token Token (CNT)", config.use_current_next_token])
    pt.add_row(["Current-Next Token Metaphone (CNM)", config.use_current_next_metaphone])

    pt.add_row(['First Token (FT)', config.use_first_token])

    print(pt)
    print("\n")

    logistic_regression_model = LogisticRegression()
    logistic_regression_model.train(training_instances)
    predictions = logistic_regression_model.predict_test_set(test_instances)

    ####################################################################################
    # This ends the baseline model code; now we just write predictions.                #
    ####################################################################################

    # check if the directory exists
    directory = os.path.dirname(config.output_predictions)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(config.output_predictions, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')


def load_data(filename, context_json, user="+H9QWAV4"):
    """
    This method loads and returns the data in filename. If the data is labelled training data, it returns labels too.

    Parameters:
        filename: the location of the training or test data you want to load.

    Returns:
        data: a list of InstanceData objects from that data type and track.
        labels (optional): if you specified training data, a dict of instance_id:label pairs.
    """

    # 'data' stores a list of 'InstanceData's as values.
    data = []
    # If this is training data, then 'labels' is a dict that contains instance_ids as keys and labels as values.
    training = False
    if filename.find('train') != -1:
        training = True

    if training:
        labels = dict()

    num_exercises = 0
    print('Loading instances...')

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()

            # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue
            if len(line) == 0:
                num_exercises += 1
                if num_exercises % 100000 == 0:
                    print('Loaded ' + str(len(data)) + ' instances across ' + str(num_exercises) + ' exercises...')

            # If the line starts with #, then we're beginning a new exercise
            elif line[0] == '#':
                list_of_exercise_parameters = line[2:].split()
                instance_properties = dict()
                for exercise_parameter in list_of_exercise_parameters:
                    [key, value] = exercise_parameter.split(':')
                    if key == 'countries':
                        value = value.split('|')
                    elif key == 'days':
                        value = float(value)
                    elif key == 'time':
                        if value == 'null':
                            value = None
                        else:
                            assert '.' not in value
                            value = int(value)
                    instance_properties[key] = value

            # Otherwise we're parsing a new Instance for the current exercise
            else:
                line = line.split()

                if training:
                    assert len(line) == 7
                else:
                    assert len(line) == 6
                assert len(line[0]) == 12

                instance_properties['instance_id'] = line[0]

                instance_properties['token'] = line[1]
                instance_properties['part_of_speech'] = line[2]

                instance_properties['morphological_features'] = dict()
                for l in line[3].split('|'):
                    [key, value] = l.split('=')
                    if key == 'Person':
                        value = int(value)
                    instance_properties['morphological_features'][key] = value

                instance_properties['dependency_label'] = line[4]
                instance_properties['dependency_edge_head'] = int(line[5])
                if training:
                    label = float(line[6])
                    labels[instance_properties['instance_id']] = label
                
                # get the instance for the context data from the context json
                exercise_id = instance_properties['instance_id'][:8]

                # get session id
                session_id = instance_properties['instance_id'][8:10]
                
                # context data from the json
                # TODO: include only the token index and the first token index
                try:
                    instance_properties['context_data'] = context_json[exercise_id][session_id]
                    data.append(InstanceData(instance_properties=instance_properties))
                except:
                    print("Not able to get data", exercise_id, session_id)


        print('Done loading ' + str(len(data)) + ' instances across ' + str(num_exercises) +
              ' exercises.\n')

    if training:
        return data, labels
    else:
        return data


class InstanceData(object):
    """
    A bare-bones class to store the included properties of each instance. This is meant to act as easy access to the
    data, and provides a launching point for deriving your own features from the data.
    """
    def __init__(self, instance_properties):

        # Parameters specific to this instance
        self.instance_id = instance_properties['instance_id']
        self.token = instance_properties['token']
        self.part_of_speech = instance_properties['part_of_speech']
        self.morphological_features = instance_properties['morphological_features']
        self.dependency_label = instance_properties['dependency_label']
        self.dependency_edge_head = instance_properties['dependency_edge_head']

        # Derived parameters specific to this instance
        self.exercise_index = int(self.instance_id[8:10])
        self.token_index = int(self.instance_id[10:12])

        # Derived parameters specific to this exercise
        self.exercise_id = self.instance_id[:10]

        # Parameters shared across the whole session
        self.user = instance_properties['user']
        self.countries = instance_properties['countries']
        self.days = instance_properties['days']
        self.client = instance_properties['client']
        self.session = instance_properties['session']
        self.format = instance_properties['format']
        self.time = instance_properties['time']

        # Derived parameters shared across the whole session
        self.session_id = self.instance_id[:8]

        # custom context data
        self.context_data = instance_properties['context_data']

    def to_features(self, config):
        """
        Prepares those features that we wish to use in the LogisticRegression example in this file. We introduce a bias,
        and take a few included features to use. Note that this dict restructures the corresponding features of the
        input dictionary, 'instance_properties'.

        Returns:
            to_return: a representation of the features we'll use for logistic regression in a dict. A key/feature is a
                key/value pair of the original 'instance_properties' dict, and we encode this feature as 1.0 for 'hot'.
        """
        to_return = dict()

        to_return['bias'] = 1.0
        to_return['user:' + self.user] = 1.0
        to_return['user:' + self.user + ':format:' + self.format] = 1.0
        to_return['token:' + self.token.lower()] = 1.0

        to_return['part_of_speech:' + self.part_of_speech] = 1.0
        for morphological_feature in self.morphological_features:
            to_return['morphological_feature:' + morphological_feature] = 1.0
        to_return['dependency_label:' + self.dependency_label] = 1.0

        to_return['session:' + self.session] = 1.0

        token_id = self.instance_id[10:12]
        token_data = self.context_data[token_id]

        if config.use_first_token:
            first_token = self.context_data['01']
            to_return['first-token:' + first_token['token']] = 1.0

        if "previous_token" in token_data:
            # if self.format == 'reverse_translate' or self.format == 'reverse_tap':

            if config.use_prev_current_metaphone:
                _token = token_data['previous_token'] + self.token
                to_return['metaphone:' + doublemetaphone(_token)[0]] = 1.0

            if config.use_prev_current_token:
                to_return['previous_token:' + token_data['previous_token'].lower()
                        + ":current_token:" + self.token.lower()] = 1.0

            if config.use_prev_current_pos:
                to_return['previous_pos:' + token_data['previous_part_of_speech']
                      + ":current_pos:" + self.part_of_speech] = 1.0
        
        # this check is used because if the token is the last token in the instance,
        # then there will be no next token
        if "next_token" in token_data:
            # in our experiments we use these features for reverse translate and reverse tap exercises
            # there was a dip in the results for listening exercises (refer our paper for more details)
            if self.format == 'reverse_translate' or self.format == 'reverse_tap':
                if config.use_current_next_metaphone:
                    _token = self.token + token_data['next_token']            
                    to_return['next-metaphone:' + doublemetaphone(_token)[0]] = 1.0
                if config.use_current_next_token:
                    to_return['next_token:' + token_data['next_token'].lower()
                        + ":current_token:" + self.token.lower()] = 1.0
                if config.use_current_next_pos:
                    to_return['next_part_of_speech:' + token_data['next_part_of_speech']
                            + ":current_pos:" + self.part_of_speech] = 1.0

        return to_return


class LogisticRegressionInstance(namedtuple('Instance', ['features', 'label', 'name'])):
    """
    A named tuple for packaging together the instance features, label, and name.
    """
    def __new__(cls, features, label, name):
        if label:
            if not isinstance(label, (int, float)):
                raise TypeError('LogisticRegressionInstance label must be a number.')
            label = float(label)
        if not isinstance(features, dict):
            raise TypeError('LogisticRegressionInstance features must be a dict.')
        return super(LogisticRegressionInstance, cls).__new__(cls, features, label, name)


class LogisticRegression(object):
    """
    An L2-regularized logistic regression object trained using stochastic gradient descent.
    """

    def __init__(self, sigma=_DEFAULT_SIGMA, eta=_DEFAULT_ETA, epochs=10):
        super(LogisticRegression, self).__init__()
        self.sigma = sigma  # L2 prior variance
        self.eta = eta  # initial learning rate
        self.epochs = epochs
        self.weights = defaultdict(lambda: uniform(-1.0, 1.0)) # weights initialize to random numbers
        self.fcounts = None # this forces smaller steps for things we've seen often before

    def predict_instance(self, instance):
        """
        This computes the logistic function of the dot product of the instance features and the weights.
        We truncate predictions at ~10^(-7) and ~1 - 10^(-7).
        """
        a = min(17., max(-17., sum([float(self.weights[k]) * instance.features[k] for k in instance.features])))
        return 1. / (1. + math.exp(-a))

    def error(self, instance):
        return instance.label - self.predict_instance(instance)

    def reset(self):
        self.fcounts = defaultdict(int)

    def training_update(self, instance):
        if self.fcounts is None:
            self.reset()
        err = self.error(instance)
        for k in instance.features:
            rate = self.eta / math.sqrt(1 + self.fcounts[k])
            # L2 regularization update
            if k != 'bias':
                self.weights[k] -= rate * self.weights[k] / self.sigma ** 2
            # error update
            self.weights[k] += rate * err * instance.features[k]
            # increment feature count for learning rate
            self.fcounts[k] += 1

    def train(self, train_set, iterations=10):
        for it in range(self.epochs):
            print('Training iteration ' + str(it+1) + '/' + str(iterations) + '...')
            shuffle(train_set)
            for instance in train_set:
                self.training_update(instance)
        print('\n')

    def predict_test_set(self, test_set):
        return {instance.name: self.predict_instance(instance) for instance in test_set}


if __name__ == '__main__':
    main()
