from logging import raiseExceptions
from matplotlib.pyplot import cla
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
# add
import data_loader_copy
from get_args import Args
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import time
import mne
from sklearn.ensemble import RandomForestClassifier

 # split dataset into test set, train set and unlabel pool

def split(args): # dataset, train_size, test_size
    # x = dataset[:, :-1]
    # y = dataset[:, -1]
    # x_train, x_pool, y_train, y_pool = train_test_split(
    #     x, y, train_size = train_size)
    # unlabel, x_test, label, y_test = train_test_split(
    #     x_pool, y_pool, test_size = test_size)
    
    
    data = data_loader_copy.Dataset(args, phase="train")
    data_pool = data_loader_copy.Dataset(args, phase="pool")        
    data_valid = data_loader_copy.Dataset(args, phase="valid")
    data_test = data_loader_copy.Dataset(args, phase="test")
    
    x_train = data.x; y_train = data.y
    x_test = data_test.x; y_test = data_test.y
    unlabel = np.concatenate((data_pool.x, data_valid.x), axis=0) 
    label = np.concatenate((data_pool.y, data_valid.y), axis=0)
    # print(label.shape, data_pool.x.shape, data_valid.x.shape)
    
    # unlabel = data_pool.x
    # label = data_pool.y

    size = data.x.shape[0] + data_pool.x.shape[0] + data_test.x.shape[0] + data_valid.x.shape[0]
    return x_train, y_train, x_test, y_test, unlabel, label, size
 
 
if __name__ == '__main__': 
    # run both models 100 times and take the average of their accuracy
    ac1, ac2 = [], []  # arrays to store accuracy of different models
    acc = []
    
    args_class = Args()
    args = args_class.args

    # time
    tm = time.localtime(time.time())
    string = time.strftime('%Y%m%d_%H%M%S', tm)

    # Save a file
    df = pd.DataFrame(columns = ['test_subj', 'acc']); idx = 0

    x_train, y_train, x_test, y_test, unlabel, label, size = split(args)
    budget = int(unlabel.shape[0]/20)

    for i in range(151):
        print(f"==={i}===")
        print(x_train.shape, unlabel.shape)

        # feature extractor 
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3) # 주성분을 몇개로 할지 결정
        printcipalComponents = pca.fit_transform(x_train)
        x_train_feature = printcipalComponents

        
        # classifier = svm.SVC(probability=True)
        # classifier = KNeighborsClassifier(n_neighbors=3)
        classifier = RandomForestClassifier()

        classifier.fit(x_train_feature,  y_train)
        printcipalComponents = pca.fit_transform(unlabel)
        unlabel_feature = printcipalComponents
        y_probab = classifier.predict_proba(unlabel_feature) #[:,0]
        # uncertainty      
        uncertainty = np.ones(len(y_probab)) - np.max(y_probab, axis=1)
        uncrt_pt_ind = np.argmax(uncertainty)
        # random
        # uncrt_pt_ind = np.random.randint(len(y_probab))

        pseudo_labeling = np.argmax(y_probab[uncrt_pt_ind])
        
        x_train = np.append(unlabel[[uncrt_pt_ind], :], x_train, axis = 0)
        y_train = np.append([pseudo_labeling], y_train)
        unlabel = np.delete(unlabel, [uncrt_pt_ind], axis = 0)
        label = np.delete(label, [uncrt_pt_ind])
        
        printcipalComponents = pca.fit_transform(x_test)
        x_test_feature = printcipalComponents
        test_acc = classifier.score(x_test_feature, y_test)
        acc.append(test_acc)
 
        df.loc[idx] = [args.test_subj, test_acc]
        df.to_csv('./csvs/{}results_choose_{}_subj{}_uncertainty.csv'.format(string, "RF", args.test_subj), header = True, index = False)
        idx += 1

        # classifier2 = LogisticRegression()
        # classifier2 = svm.SVC(probability=True)
        # classifier2 = KNeighborsClassifier(n_neighbors=3)
        classifier2 = RandomForestClassifier()
        classifier2.fit(x_test, y_test)
        
        # train_size = x_train.shape[0]/size
        # x_train, y_train, x_test, y_test, unlabel, label, size = split()
 
        # train model without active learning
        # classifier3 = LogisticRegression()
        # classifier3 = svm.SVC(probability=True)
        # classifier3.fit(x_train, y_train)
        # ac2.append(classifier3.score(x_test, y_test))    
    # figure
    from matplotlib import pyplot as plt
    plt.plot(np.arange(len(acc)), acc)
    plt.savefig(f"./{string}_svm.png")

# random도 하고.
