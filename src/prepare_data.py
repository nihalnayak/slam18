"""
This program is used to generate a csv that captures instance level feature values for the next and previous token
in a given exercise.We also capture the POS of the tokens. 
"""

import argparse
import json
import csv
import os
from config import Config


def main():
    parser = argparse.ArgumentParser(description='Duolingo shared task. Compute additional features')
    parser.add_argument('--params_file', required=True, help='The parameters file with all the options')
    args = parser.parse_args()
    config = Config(args.params_file)

    # the train file
    print("Preparing data for the Training Set")
    load_and_compute(config.train_file)

    # the dev file
    print("Preparing data for the Dev Set")
    print(config.dev_file)
    load_and_compute(config.dev_file)


    # the test file
    print("Preparing data for the Test Set")
    load_and_compute(config.test_file)

    print("Process successfully completed!")
    print("Run the following the command - ")
    print("python train_model.py --params_file parameters.ini")


def load_and_compute(file_path, user="+H9QWAV4"):
    data = []

    # token informations

    tokens_data = {}

    # If this is training data, then 'labels' is a dict that contains instance_ids as keys and labels as values.
    num_exercises = 0
    number_of_instances = 0
    print('Loading instances...')

    with open(file_path, 'rt') as f:
        user_exercise = ""
        for line in f:
            line = line.strip()
            # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue
            if len(line) == 0:
                num_exercises += 1
                user_exercise = ""
                if num_exercises % 100000 == 0:
                    print('Loaded ' + str(number_of_instances) + ' instances across ' + str(num_exercises) + ' exercises...')

            # If the line starts with #, then we're beginning a new exercise
            elif line[0] == '#':
                list_of_exercise_parameters = line[2:].split()
                for exercise_parameter in list_of_exercise_parameters:
                    [key, value] = exercise_parameter.split(':')

            # Otherwise we're parsing a new Instance for the current exercise
            else:
                line = line.split()

                # instance id
                instance_id = line[0]

                token = line[1]
                part_of_speech = line[2]

                # get the the exercise id
                exercise_id = instance_id[:8]

                # get session id
                session_id = instance_id[8:10]

                # token index
                token_id = instance_id[10:12]

                if exercise_id in tokens_data:
                    if session_id in tokens_data[exercise_id]:
                        # get the previous value
                        previous_token_id = "%02d" % (int(token_id) - 1)
                        previous_dict = tokens_data[exercise_id][session_id][previous_token_id]

                        # update the present token's data with previous data
                        tokens_data[exercise_id][session_id][token_id] = {
                            'token': token,
                            'part_of_speech': part_of_speech,
                            'previous_token': previous_dict['token'],
                            'previous_part_of_speech': previous_dict['part_of_speech']
                        }

                        # update the previous dictionary with the current value
                        tokens_data[exercise_id][session_id][previous_token_id]['next_token'] = token
                        tokens_data[exercise_id][session_id][previous_token_id]['next_part_of_speech'] = part_of_speech

                    else:
                        tokens_data[exercise_id][session_id] = {
                            token_id: {
                                "token": token,
                                "part_of_speech": part_of_speech
                            }
                        }

                else:
                    # create exercise id

                    tokens_data[exercise_id] = {
                        session_id: {
                            token_id: {
                                "token": token,
                                "part_of_speech": part_of_speech
                            }
                        }
                    }

                number_of_instances += 1

        print('Done loading ' + str(number_of_instances) + ' instances across ' + str(num_exercises) +
              ' exercises.\n')

        # dump the data as a json
        with open(file_path + ".json", "w") as token_json:
            json.dump(tokens_data, token_json)

        f.close()

if __name__ == "__main__":
    main()
