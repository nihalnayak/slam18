"""
Duolingo SLAM Shared Task - Baseline Model

This baseline model loads the training and test data that you pass in via --train and --test arguments for a particular
track (course), storing the resulting data in InstanceData objects, one for each instance. The code then creates the
features we'll use for logistic regression, storing the resulting LogisticRegressionInstance objects, then uses those to
train a regularized logistic model with SGD, and then makes predictions for the test set and dumps them to a CSV file
specified with the --pred argument, in a format appropriate to be read in and graded by the eval.py script.

We elect to use two different classes, InstanceData and LogisticRegressionInstance, to delineate the boundary between
the two purposes of this code; the first being to act as a user-friendly interface to the data, and the second being to
train and run a baseline model as an example. Competitors may feel free to use InstanceData in their own code, but
should consider replacing the LogisticRegressionInstance with a class more appropriate for the model they construct.

This code is written to be compatible with both Python 2 or 3, at the expense of dependency on the future library. This
code does not depend on any other Python libraries besides future.
"""

import argparse
from collections import defaultdict, namedtuple
from io import open
import json
import math
import os
from random import shuffle, uniform

from future.builtins import range
from future.utils import iteritems
from metaphone import doublemetaphone

# Sigma is the L2 prior variance, regularizing the baseline model. Smaller sigma means more regularization.
_DEFAULT_SIGMA = 30.0

# Eta is the learning rate/step size for SGD. Larger means larger step size.

_DEFAULT_ETA = 0.40

previous_next_tokens_json = {}


def main():
    global previous_next_tokens_json
    """
    Executes the baseline model. This loads the training data, training labels, and dev data, then trains a logistic
    regression model, then dumps predictions to the specified file.

    Modify the middle of this code, between the two commented blocks, to create your own model.
    """

    parser = argparse.ArgumentParser(description='Duolingo shared task baseline model')
    parser.add_argument('--train', help='Training file name', required=True)
    parser.add_argument('--test', help='Test file name, to make predictions on', required=True)
    # parser.add_argument('--dev', help='dev file, to make predictions on', required=True)
    # parser.add_argument('--dev_key', help='Dev key, to make predictions on', required=True)
    parser.add_argument('--pred', help='Output file name for predictions, defaults to test_name.pred')
    args = parser.parse_args()

    if not args.pred:
        args.pred = args.test + 'paper.baseline.pcpos_cnpos.pcm_cnm.pct_cnt.ft.pred'

    assert os.path.isfile(args.train)
    assert os.path.isfile(args.test)

    # Assert that the train course matches the test course
    # assert os.path.basename(args.train)[:5] == os.path.basename(args.test)[:5]

    with open(args.train + ".json") as token_json:
        train_token_json = json.load(token_json)

    with open(args.test + ".json") as token_json:
        test_token_json = json.load(token_json)

    previous_next_tokens_json = dict(train_token_json, **test_token_json)

    training_data, training_labels = load_data(args.train)
    # dev_data = load_data(args.dev)
    # dev_labels = load_labels(args.dev_key)

    test_data = load_data(args.test)

    ####################################################################################
    # Here is the delineation between loading the data and running the baseline model. #
    # Replace the code between this and the next comment block with your own.          #
    ####################################################################################

    training_instances = [LogisticRegressionInstance(features=instance_data.to_features(),
                                                     label=training_labels[instance_data.instance_id],
                                                     name=instance_data.instance_id
                                                     ) for instance_data in training_data]

    # dev_instances = [LogisticRegressionInstance(features=instance_data.to_features(),
    #                                             label=dev_labels[instance_data.instance_id],
    #                                             name=instance_data.instance_id
    #                                             ) for instance_data in dev_data]

    # training_instances += dev_instances

    test_instances = [LogisticRegressionInstance(features=instance_data.to_features(),
                                                 label=None,
                                                 name=instance_data.instance_id
                                                 ) for instance_data in test_data]

    logistic_regression_model = LogisticRegression()
    logistic_regression_model.train(training_instances, iterations=10)

    predictions = logistic_regression_model.predict_test_set(test_instances)

    ####################################################################################
    # This ends the baseline model code; now we just write predictions.                #
    ####################################################################################

    with open(args.pred, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def load_labels(filename):
    """
    This loads labels, either the actual ones or your predictions.

    Parameters:
        filename: the filename pointing to your labels

    Returns:
        labels: a dict of instance_ids as keys and labels between 0 and 1 as values
    """
    labels = dict()

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                line = line.split()
            instance_id = line[0]
            label = float(line[1])
            labels[instance_id] = label
    return labels


def load_data(filename):
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
                user_exercise = ""
                if num_exercises % 100000 == 0:
                    print('Loaded ' + str(len(data)) + ' instances across ' + str(num_exercises) + ' exercises...')

            # If the line starts with #, then we're beginning a new exercise
            elif line[0] == '#':
                list_of_exercise_parameters = line[2:].split()
                instance_properties = dict()
                for exercise_parameter in list_of_exercise_parameters:
                    [key, value] = exercise_parameter.split(':')
                    # if key == 'user':
                    #     user_exercise = value
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
                # if user_exercise.strip() != user.strip():
                #     continue
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
                data.append(InstanceData(instance_properties=instance_properties))

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

    global previous_next_tokens_json

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

    def to_features(self):
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
        # to_return['length_of_token'] = float(len(self.token))
        to_return['user:' + self.user] = 1.0

        to_return['user:'+self.user+':format:' + self.format] = 1.0
        to_return['session:' + self.session] = 1.0
        to_return['token:' + self.token.lower()] = 1.0

        to_return['part_of_speech:' + self.part_of_speech] = 1.0
        for morphological_feature in self.morphological_features:
            to_return['morphological_feature:' + morphological_feature] = 1.0
        to_return['dependency_label:' + self.dependency_label] = 1.0

        #if is_ascii(self.token):
        #    to_return['accent:'+self.user] = 1.0

        # if self.format == 'reverse_translate' or self.format == 'reverse_tap':
        exercise_id = self.instance_id[:8]

        # get session id
        session_id = self.instance_id[8:10]

        # token index
        token_id = self.instance_id[10:12]

        # to_return['']

        # get token data

        first_token = previous_next_tokens_json[exercise_id][session_id]['01']

        to_return['first-token:' + first_token['token']] = 1.0

        token_data = previous_next_tokens_json[exercise_id][session_id][token_id]

        if "previous_token" in token_data:
            # if self.format == 'reverse_translate' or self.format == 'reverse_tap':

            _token = token_data['previous_token'] + self.token
            to_return['metaphone:' + doublemetaphone(_token)[0]] = 1.0

            to_return['previous_token:' + token_data['previous_token'].lower()
                      + ":current_token:" + self.token.lower()] = 1.0

            to_return['previous_pos:' + token_data['previous_part_of_speech']
                      + ":current_pos:" + self.part_of_speech] = 1.0

            # else:
            #     _token = token_data['previous_token'] + self.token
            #     to_return['metaphone:' + doublemetaphone(_token)[0]] = 1.0
                # _token = doublemetaphone(token_data['previous_token'])[0][-1] + doublemetaphone(self.token)[0][0]
                # to_return['metaphone:' + _token] = 1.0

        if "next_token" in token_data:
            _token = self.token + token_data['next_token']
            # to_return['next-metaphone:' + doublemetaphone(_token)[0]] = 1.0
            if self.format == 'reverse_translate' or self.format == 'reverse_tap':
                to_return['next-metaphone:' + doublemetaphone(_token)[0]] = 1.0
                to_return['next_token:' + token_data['next_token'].lower()
                      + ":current_token:" + self.token.lower()] = 1.0
                # to_return['next_part_of_speech:' + token_data['next_part_of_speech']] = 1.0
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

    def __init__(self, sigma=_DEFAULT_SIGMA, eta=_DEFAULT_ETA):
        super(LogisticRegression, self).__init__()
        self.sigma = sigma  # L2 prior variance
        self.eta = eta  # initial learning rate
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

    def train(self, train_set, iterations=20):
        for it in range(iterations):
            print('Training iteration ' + str(it+1) + '/' + str(iterations) + '...')
            shuffle(train_set)
            for instance in train_set:
                self.training_update(instance)
        print('\n')

    def predict_test_set(self, test_set):
        return {instance.name: self.predict_instance(instance) for instance in test_set}


if __name__ == '__main__':
    main()
