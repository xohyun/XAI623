'''
use https://github.com/modAL-python/modAL
'''

from sys import stderr
# from scipy._lib.six import X
from scipy.signal.windows.windows import exponential
from torch.optim import optimizer
import torch
import data_loader
from modAL.models.learners import Committee
import trainer
import os
import pandas as pd
from get_args import Args
import subprocess
from utils import *
# from Model.model_maker import ModelMaker
from trainer import TrainMaker
###
import numpy as np

from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, uncertainty_sampling

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from IPython import display
from matplotlib import pyplot as plt

def main():
    args_class = Args()
    args = args_class.args
 
    # Fix seed
    if args.seed:
        fix_random_seed(args)
    
    # Save a file 
    df = pd.DataFrame(columns = ['test_subj', 'lr', 'wd', 'acc', 'f1', 'loss']); idx = 0

    # Load data
    data = data_loader.Dataset(args, phase="train")
    valid_data = data_loader.Dataset(args, phase="valid")

    # Prepare folder
    prepare_folder(args.param_path, args.runs_path)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)

    n_members = 5
    learner_list = list()

    for member_idx in range(n_members):
        n_initial = 3
        train_index = np.random.choice(range(valid_data.x.shape[0]), size=n_initial, replace=False)
        
        data.x, data.y = valid_data.x[train_index], valid_data.y[train_index]
        valid_data.x, valid_data.y = np.delete(valid_data.x, train_index, axis=0), np.delete(valid_data.y, train_index)

        learner = ActiveLearner(
            # estimator=RandomForestClassifier(),
            estimator=knn,
            # query_strategy=uncertainty_sampling,
            query_strategy=entropy_sampling,
            X_training=data.x, y_training=data.y
        )
        learner_list.append(learner)
    committee = Committee(learner_list=learner_list)
    
    N_QUERIES = 50

    for index in range(N_QUERIES):
        query_index, query_instance = committee.query(valid_data.x)
        committee.teach(X=valid_data.x[query_index], y=valid_data.y[query_index])
        valid_data.x, valid_data.y = np.delete(valid_data.x, query_index, axis=0), np.delete(valid_data.y, query_index)
        # print("++++++++++++++", valid_data.x.shape)
        model_accuracy = learner.score(data.x, data.y)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

if __name__ == "__main__" :
    main()