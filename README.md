# Context based Approach for Second Language Acquisition
This project is the implementation of the system submitted to the SLAM 2018 (Second Language Acquisition Modeling 2018) shared task.

This page gives instructions for replicating the results in our system.

## Table of Contents

<!-- toc -->

- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Downloading Data](#downloading-data)
- [Parameters for the Experiment](#parameters-for-the-experiment)
- [Prepare Data for training](#prepare-data)
- [Train your model](#train-your-model)
- [Test your predictions](#test-your-predictions)
- [Citation](#citation)

<!-- tocstop -->

## Installation
Our project is built on python. We have ensured python 2 and 3 compatibility. In this section,
we describe the installation procedure for Ubuntu 16.04.

```shell
git clone https://github.com/iampuntre/slam18.git
cd slam18
virtualenv env
source env/bin/activate
pip install -r requirements.txt
mkdir data
```

Note: Follow equivalent instructions for your Operating System

## Downloading Data
In our experiments, we use the SLAM 2018 Dataset. To download the dataset, download from [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO](here). 

After downloading the data, unzip it in the ``data`` directory.

## Parameters for the Experiment
In order to train the model, you will have to configure the `parameters.ini` file. You can find this file in `src/parameters.ini`.

We have three sections in the file - `model`, `options` and `context_features`. The `model` section is used to point to the train and test files. `options` section is used to manipulate the various hyperparameters while training the model. Lastly, we have the `context_features` section, which is used to activate/deactivate various context based features.

Change the appropriate values for train, dev and test files. We have preset the values of the hyperparameters that we have used in our experiments. By default, all the context features are activated.

## Prepare Data
After you have successfully configured the `parameters.ini` file, you should prepare the data for training. This is an intermediate step, where we extract the tokens and part of speech present in the surrounding of the context. For more details, read our paper. 

To prepare the data, execute the following - 

```shell
python src/prepare_data.py --params_file src/parameters.ini
```

You should be able to see three `.json` files in your data directory.

## Train your model 
To train the model, type the following command in your terminal - 

```shell
python src/train_model.py --params_file src/parameters.ini
```

Note: It is recommended you run this step only if you have sufficient memory (atleast 16GB)

## Test your predictions
To evaluate your predictions, execute the following command - 

```shell
python src/eval.py --params_file src/parameters.ini
```

## Citation
If you make use of this work in your paper, please cite our paper 

```
@InProceedings{nayak-rao:2018:W18-05,
  author    = {Nayak, Nihal V.  and  Rao, Arjun R.},
  title     = {Context Based Approach for Second Language Acquisition},
  booktitle = {Proceedings of the Thirteenth Workshop on Innovative Use of NLP for Building Educational Applications},
  month     = {June},
  year      = {2018},
  address   = {New Orleans, Louisiana},
  publisher = {Association for Computational Linguistics},
  pages     = {212--216},
  url       = {http://www.aclweb.org/anthology/W18-0524}
}


```

