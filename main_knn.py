'''
use https://github.com/modAL-python/modAL
'''

from sys import stderr
# from scipy._lib.six import X
from scipy.signal.windows.windows import exponential
from torch.optim import optimizer
import torch
import data_loader
import trainer
import os
import pandas as pd
from get_args import Args
import subprocess
from utils import *
from trainer import TrainMaker
import numpy as np

from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, uncertainty_sampling

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from IPython import display
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def main():
    args_class = Args()
    args = args_class.args
 
    # Fix seed
    if args.seed:
        fix_random_seed(args)
    
    # Save a file 
    # df = pd.DataFrame(columns = ['test_subj', 'lr', 'wd', 'acc', 'f1', 'loss']); idx = 0
    df = pd.DataFrame(columns = ['acc']); idx = 0

    # Load data
    data = data_loader.Dataset(args, phase="train")
    valid_data = data_loader.Dataset(args, phase="valid")

    # Prepare folder
    prepare_folder([args.param_path, args.runs_path])

    knn = KNeighborsClassifier(n_neighbors=3)
    learner = ActiveLearner(
        estimator=knn,
        query_strategy=uncertainty_sampling,
        # query_strategy=entropy_sampling,
        X_training=data.x, y_training=data.y
    )
    
    N_QUERIES = 150
    performance_history = [] #[unqueried_score]

    for index in range(N_QUERIES):
        query_index, query_instance = learner.query(valid_data.x)
        learner.teach(X=valid_data.x[query_index], y=valid_data.y[query_index])
        valid_data.x, valid_data.y = np.delete(valid_data.x, query_index, axis=0), np.delete(valid_data.y, query_index)
        # print("++++++++++++++", valid_data.x.shape)
        model_accuracy = learner.score(data.x, data.y)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

        performance_history.append(model_accuracy)
        df.loc[idx] = [model_accuracy]
    
    plt.plot(np.arange(1,N_QUERIES+1), performance_history)
    plt.savefig("./knn2.png")
    print(performance_history)
    # df.loc[idx] = [args.test_subj, args.lr, args.wd, acc_v, f1_v, loss_v.cpu().numpy()]
    # df.to_excel('results_{}_subj{}.xlsx'.format(args.model, args.test_subj), header = True, index = False)

if __name__ == "__main__" :
    main()
