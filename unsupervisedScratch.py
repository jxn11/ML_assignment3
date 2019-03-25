#### Attribution of source code used:
# Learning Curve: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
# ROC graph: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
# Confusion matrix: https://gist.github.com/zachguo/10296432
# Probability for LinearSVC: https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from sklearn import preprocessing

import math

import os

import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import tree
import librosa
import librosa.display
import graphviz
import pickle
import utils
import os.path

from pprint import pprint
from time import time
import logging

from sklearn.externals.six.moves import zip
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import print_cm
import confusionMatrix

from sklearn.metrics import confusion_matrix

import scikitplot as skplt
import matplotlib.pyplot as plt

from numpy import genfromtxt
import random

import time

from Solid.GeneticAlgorithm import GeneticAlgorithm
from Solid.StochasticHillClimb import StochasticHillClimb
from Solid.SimulatedAnnealing import SimulatedAnnealing

import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn import metrics
from sklearn.mixture import GaussianMixture
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import random_projection
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

from sklearn.preprocessing import OneHotEncoder

# Toggle dataset
fma = False
bna = True

# Toggle Cluster type
kmeans = False
expectMax = True

# part 2
pComp = False

# part 3
clust2 = False

# part 4 & 5
part4 = False
part5 = True
neuralNet_paramSearch = True
neuralNet_learningCurve = True
neuralNet_testAssess = True

def convertToStr_bna(array):

    # convert num labels to strings
    remappedArray = []
    for label in array:
        if label == 0:
            remappedArray.append('Genuine')
        elif label == 1:
            remappedArray.append('Forged')
    return remappedArray


### DATA PROCESSING ############################################################

if fma:
    # Paths and files
    audio_dir = '../../data/fma_metadata/'
    localDataFile = 'trainTestData.pkl'

    if os.path.exists(localDataFile):
        with open(localDataFile, 'rb') as f:
            data = pickle.load(f)
        y_train = data[0]; y_val = data[1]; y_test = data[2]
        X_train = data[3]; X_val = data[4]; X_test = data[5]
    else:
        # Load metadata and features
        tracks = utils.load(audio_dir + 'tracks.csv')
        genres = utils.load(audio_dir + 'genres.csv')
        features = utils.load(audio_dir + 'features.csv')
        echonest = utils.load(audio_dir + 'echonest.csv')

        np.testing.assert_array_equal(features.index, tracks.index)
        assert echonest.index.isin(tracks.index).all()

        # Setup train/test split
        small = tracks['set', 'subset'] <= 'small'

        train = tracks['set', 'split'] == 'training'
        val = tracks['set', 'split'] == 'validation'
        test = tracks['set', 'split'] == 'test'

        y_train = tracks.loc[small & train, ('track', 'genre_top')]
        y_val = tracks.loc[small & val, ('track', 'genre_top')]
        y_test = tracks.loc[small & test, ('track', 'genre_top')]
        # X_train = features.loc[small & train, 'mfcc'] #just mfcc features
        # X_test = features.loc[small & test, 'mfcc']
        X_train = features.loc[small & train] #all audio-extracted features
        X_val = features.loc[small & val]
        X_test = features.loc[small & test]

        print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
        print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))

        # Be sure training samples are shuffled.
        X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

        # Standardize features by removing the mean and scaling to unit variance.
        scaler = skl.preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(X_train)
        scaler.transform(X_test)
        scaler.transform(X_val)

        # Save the formatted data:
        with open(localDataFile, 'wb') as f:
            # pickle.dump([y_train, y_test, X_train, X_test], f)
            pickle.dump([y_train, y_val, y_test, X_train, X_val, X_test], f)

if bna:
    # Paths and files
    bna_dir = '../../data/banknoteAuthentication/'
    localDataFile = 'trainTestData_bna.pkl'

    # load data
    bnaData = genfromtxt(bna_dir + 'banknoteAuthentication.txt', delimiter=',')

    # pos negative split
    negExamp = bnaData[:762,:]
    posExamp = bnaData[762:,:]

    # balance data
    negExamp = negExamp[:610,:]

    #shuffle examples
    np.random.shuffle(negExamp)
    np.random.shuffle(posExamp)

    X_train = np.vstack((negExamp[:488,:-1], posExamp[:488,:-1]))
    y_train = np.hstack((negExamp[:488,-1], posExamp[:488,-1]))

    X_val = np.vstack((negExamp[488:549,:-1], posExamp[488:549,:-1]))
    y_val = np.hstack((negExamp[488:549,-1], posExamp[488:549,-1]))

    X_test = np.vstack((negExamp[549:,:-1], posExamp[549:,:-1]))
    y_test = np.hstack((negExamp[549:,-1], posExamp[549:,-1]))

    # Be sure training samples are shuffled.
    X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance.
    scaler = skl.preprocessing.StandardScaler(copy=False)
    scaler.fit_transform(X_train)
    scaler.transform(X_test)

# K-Means Clustering
# check here for performance evaluation: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
if kmeans:
    if bna:

        maxClust = 20

        ariScore = np.zeros(maxClust-1)
        vmScore = np.zeros(maxClust-1)
        silScoreEuc = np.zeros(maxClust-1)
        silScoreMan = np.zeros(maxClust-1)
        silScoreMah = np.zeros(maxClust-1)

        for iComps in range(2,maxClust+1):
            print('clustNum = ' + str(iComps) + ' out of ' + str(maxClust))

            kClustering = KMeans(n_clusters=iComps, random_state=0).fit(X_train, y_train)
            ariScore[iComps-2] = metrics.adjusted_rand_score(y_train, kClustering.labels_)
            vmScore[iComps-2] = metrics.v_measure_score(y_train, kClustering.labels_)
            silScoreEuc[iComps-2] = metrics.silhouette_score(X_train, kClustering.labels_, metric='euclidean')
            silScoreMan[iComps-2] = metrics.silhouette_score(X_train, kClustering.labels_, metric='manhattan')
            silScoreMah[iComps-2] = metrics.silhouette_score(X_train, kClustering.labels_, metric='mahalanobis')
        # print(ariScore)
        # quit()

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True)

        ax1.plot(list(range(2,maxClust+1)), ariScore, linewidth=2.5)
        # ax1.set_xticks(list(range(2,maxClust+1)))
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('ARI Score')
        ax1.set_title('Adjusted Rand Index')
        ax1.legend(['ARI'])

        ax2.plot(list(range(2,maxClust+1)), vmScore, linewidth=2.5)
        # ax2.set_xticks(list(range(2,maxClust+1)))
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('V-Measure')
        ax2.set_title('V-Measure')
        ax2.legend(['V-Measure'])

        ax3.plot(list(range(2,maxClust+1)), silScoreEuc, linewidth=2.5)
        ax3.plot(list(range(2,maxClust+1)), silScoreMan, linewidth=2.5)
        ax3.plot(list(range(2,maxClust+1)), silScoreMah, linewidth=2.5)
        # ax3.set_xticks(list(range(2,maxClust+1)))
        ax3.set_xlabel('Number of Clusters')
        ax3.set_ylabel('Silhouette Coefficient')
        ax3.set_title('Silhouette Score')
        ax3.legend(['Euclidean', 'Manhattan', 'Mahalanobis'])

        plt.suptitle('BNA - K-Means Clustering Evaluation')
        plt.show()
        quit()


        kClustering = KMeans(n_clusters=2, random_state=0).fit(X_train, y_train)

        # assess the clustering performance with Adjusted Rand Index
        ari = metrics.adjusted_rand_score(y_train, kClustering.labels_)

        # reduce dimensionality of data for visualization
        embedding = MDS(n_components=2, verbose=10)
        # X_transformed = embedding.fit_transform(X_train)
        X_transformed = embedding.fit_transform(np.vstack((X_train, kClustering.cluster_centers_)))
        centroid2D = X_transformed[-2:,:]
        X_transformed = X_transformed[:-2,:]

        #convert predicitons and true labels to colors for visualzation
        cmap = cm.Set1
        predictColor = []
        trueColor = []
        for i in range(len(y_train)):
            if kClustering.labels_[i] == 0:
                predictColor.extend([cmap(1)])
            elif kClustering.labels_[i] == 1:
                predictColor.extend([cmap(4)])

            if y_train[i] == 0:
                trueColor.extend([cmap(1)])
            elif y_train[i] == 1:
                trueColor.extend([cmap(4)])


        f = plt.figure()
        plt.scatter(X_transformed[:,0], X_transformed[:,1], c = predictColor, \
                    edgecolors=trueColor)
        plt.scatter(centroid2D[0,0], centroid2D[0,1], s=400, linewidth=3, \
                    marker = 'X', color = cmap(1), edgecolors='k')
        plt.scatter(centroid2D[1,0], centroid2D[1,1], s=400, linewidth=3, \
                    marker = 'X', color = cmap(4), edgecolors='k')
        plt.title('Kmeans Clustering\nARI = ' + str(ari))
        plt.show()


        # version with "optimal" 5 clusters
        kClustering = KMeans(n_clusters=5, random_state=0).fit(X_train, y_train)

        # assess the clustering performance with Adjusted Rand Index
        ari = metrics.adjusted_rand_score(y_train, kClustering.labels_)

        # reduce dimensionality of data for visualization
        embedding = MDS(n_components=2, verbose=10)
        # X_transformed = embedding.fit_transform(X_train)
        X_transformed = embedding.fit_transform(np.vstack((X_train, kClustering.cluster_centers_)))
        centroid2D = X_transformed[-5:,:]
        X_transformed = X_transformed[:-5,:]

        #convert predicitons and true labels to colors for visualzation
        cmap = cm.Set1
        cmap2 = cm.tab10
        predictColor = []
        trueColor = []
        for i in range(len(y_train)):
            predictColor.extend([cmap(kClustering.labels_[i])])
            trueColor.extend([cmap2(y_train[i])])


        f = plt.figure()
        plt.scatter(X_transformed[:,0], X_transformed[:,1], c = predictColor, \
                    edgecolors=trueColor)
        for i in range(centroid2D.shape[0]):
            plt.scatter(centroid2D[i,0], centroid2D[i,1], s=400, linewidth=3, \
                        marker = 'X', color = cmap(i), edgecolors='k')
        plt.title('Kmeans Clustering (5 Clusters)\nARI = ' + str(ari))
        plt.show()

    if fma:

        # maxClust = 100

        # ariScore = np.zeros(maxClust-1)
        # vmScore = np.zeros(maxClust-1)
        # silScoreEuc = np.zeros(maxClust-1)
        # silScoreMan = np.zeros(maxClust-1)
        # silScoreMah = np.zeros(maxClust-1)

        # print('check')

        # for iComps in range(2,maxClust+1):
        #     print('clustNum = ' + str(iComps) + ' out of ' + str(maxClust))

        #     # kClustering = KMeans(n_clusters=iComps, random_state=0, verbose=1, n_jobs=-1).fit(X_train, y_train)
        #     kClustering = KMeans(n_clusters=8, random_state=0).fit(X_train, y_train)
        #     ariScore[iComps-2] = metrics.adjusted_rand_score(y_train, kClustering.labels_)
        #     vmScore[iComps-2] = metrics.v_measure_score(y_train, kClustering.labels_)
        #     silScoreEuc[iComps-2] = metrics.silhouette_score(X_train, kClustering.labels_, metric='euclidean')
        #     silScoreMan[iComps-2] = metrics.silhouette_score(X_train, kClustering.labels_, metric='manhattan')
        #     silScoreMah[iComps-2] = metrics.silhouette_score(X_train, kClustering.labels_, metric='mahalanobis')
        # print(ariScore)
        # quit()

        # ari = []
        # vm = []
        # silEuc = []
        # silMan = []
        # silMah = []

        # kClustering = KMeans(n_clusters=2, random_state=0).fit(X_train, y_train)
        # ari.append(metrics.adjusted_rand_score(y_train, kClustering.labels_))
        # vm.append(metrics.v_measure_score(y_train, kClustering.labels_))
        # silEuc.append(metrics.silhouette_score(X_train, kClustering.labels_, metric='euclidean'))
        # silMan.append(metrics.silhouette_score(X_train, kClustering.labels_, metric='manhattan'))
        # silMah.append(metrics.silhouette_score(X_train, kClustering.labels_, metric='mahalanobis'))

        # print(ari)

        # maxClust = 2

        # fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True)

        # ax1.plot(list(range(2,maxClust+1)), ariScore, linewidth=2.5)
        # # ax1.set_xticks(list(range(2,maxClust+1)))
        # ax1.set_xlabel('Number of Clusters')
        # ax1.set_ylabel('ARI Score')
        # ax1.set_title('Adjusted Rand Index')
        # ax1.legend(['ARI'])

        # ax2.plot(list(range(2,maxClust+1)), vmScore, linewidth=2.5)
        # # ax2.set_xticks(list(range(2,maxClust+1)))
        # ax2.set_xlabel('Number of Clusters')
        # ax2.set_ylabel('V-Measure')
        # ax2.set_title('V-Measure')
        # ax2.legend(['V-Measure'])

        # ax3.plot(list(range(2,maxClust+1)), silScoreEuc, linewidth=2.5)
        # ax3.plot(list(range(2,maxClust+1)), silScoreMan, linewidth=2.5)
        # ax3.plot(list(range(2,maxClust+1)), silScoreMah, linewidth=2.5)
        # # ax3.set_xticks(list(range(2,maxClust+1)))
        # ax3.set_xlabel('Number of Clusters')
        # ax3.set_ylabel('Silhouette Coefficient')
        # ax3.set_title('Silhouette Score')
        # ax3.legend(['Euclidean', 'Manhattan', 'Mahalanobis'])

        # plt.suptitle('BNA - K-Means Clustering Evaluation')
        # plt.show()
        # quit()

        numClasses = 8

        # create encoder to map from string to int labels
        le = preprocessing.LabelEncoder()
        le.fit(y_train.iloc[:].values)
        intLabels = le.transform(y_train.iloc[:].values)
        # print(intLabels)
        # quit()

        kClustering = KMeans(n_clusters=numClasses, random_state=0).fit(X_train, y_train)

        # assess the clustering performance with Adjusted Rand Index
        ari = metrics.adjusted_rand_score(y_train, kClustering.labels_)

        # reduce dimensionality of data for visualization
        embedding = MDS(n_components=2, verbose=10, max_iter=2000, n_jobs=-1)
        X_transformed = embedding.fit_transform(np.vstack((X_train, kClustering.cluster_centers_)))
        centroid2D = X_transformed[-numClasses:,:]
        X_transformed = X_transformed[:-numClasses,:]

        #convert predicitons and true labels to colors for visualzation
        cmap = cm.Set1
        predictColor = []
        trueColor = []
        for i in range(len(y_train)):
            predictColor.extend([cmap(kClustering.labels_[i])])
            trueColor.extend([cmap(intLabels[i])])

        f = plt.figure()
        plt.scatter(X_transformed[:,0], X_transformed[:,1], c = predictColor, \
                    edgecolors=trueColor)
        for i in range(centroid2D.shape[0]):
            plt.scatter(centroid2D[i,0], centroid2D[i,1], s=400, linewidth=3, \
                        marker = 'X', color = cmap(i), edgecolors='k')
        plt.title('Kmeans Clustering\nARI = ' + str(ari))
        plt.show()

# Expectation Maximization

if expectMax:
    if bna:

        # emClustering = GaussianMixture(n_components=2, covariance_type='full', \
        #                                max_iter=100).fit(X_train, y_train)
        # emPredictions = emClustering.predict(X_train)

        # # assess the clustering performance with Adjusted Rand Index
        # ari = metrics.adjusted_rand_score(y_train, emPredictions)

        # # reduce dimensionality of data for visualization
        # embedding = MDS(n_components=2, verbose=10)
        # X_transformed = embedding.fit_transform(np.vstack((X_train, emClustering.means_)))
        # centroid2D = X_transformed[-2:,:]
        # X_transformed = X_transformed[:-2,:]

        # #convert predicitons and true labels to colors for visualzation
        # cmap = cm.Set1
        # predictColor = []
        # trueColor = []
        # for i in range(len(y_train)):
        #     if emPredictions[i] == 0:
        #         predictColor.extend([cmap(1)])
        #     elif emPredictions[i] == 1:
        #         predictColor.extend([cmap(4)])

        #     if y_train[i] == 0:
        #         trueColor.extend([cmap(1)])
        #     elif y_train[i] == 1:
        #         trueColor.extend([cmap(4)])


        # f = plt.figure()
        # plt.scatter(X_transformed[:,0], X_transformed[:,1], c = predictColor, \
        #             edgecolors=trueColor)
        # plt.scatter(centroid2D[0,0], centroid2D[0,1], s=400, linewidth=3, \
        #             marker = 'X', color = cmap(1), edgecolors='k')
        # plt.scatter(centroid2D[1,0], centroid2D[1,1], s=400, linewidth=3, \
        #             marker = 'X', color = cmap(4), edgecolors='k')
        # plt.title('Expectation Maximization Clustering\nARI = ' + str(ari))
        # plt.show()


        # maxClust = 20

        # ariScore = np.zeros(maxClust-1)
        # vmScore = np.zeros(maxClust-1)
        # silScoreEuc = np.zeros(maxClust-1)
        # silScoreMan = np.zeros(maxClust-1)
        # silScoreMah = np.zeros(maxClust-1)

        # for iComps in range(2,maxClust+1):
        #     print('clustNum = ' + str(iComps) + ' out of ' + str(maxClust))

        #     emClustering = GaussianMixture(n_components=iComps, covariance_type='full', \
        #                                max_iter=100).fit(X_train, y_train)
        #     emPredictions = emClustering.predict(X_train)

        #     ariScore[iComps-2] = metrics.adjusted_rand_score(y_train, emPredictions)
        #     vmScore[iComps-2] = metrics.v_measure_score(y_train, emPredictions)
        #     silScoreEuc[iComps-2] = metrics.silhouette_score(X_train, emPredictions, metric='euclidean')
        #     silScoreMan[iComps-2] = metrics.silhouette_score(X_train, emPredictions, metric='manhattan')
        #     silScoreMah[iComps-2] = metrics.silhouette_score(X_train, emPredictions, metric='mahalanobis')
        # # print(ariScore)
        # # quit()

        # fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True)

        # ax1.plot(list(range(2,maxClust+1)), ariScore, linewidth=2.5)
        # # ax1.set_xticks(list(range(2,maxClust+1)))
        # ax1.set_xlabel('Number of Clusters')
        # ax1.set_ylabel('ARI Score')
        # ax1.set_title('Adjusted Rand Index')
        # ax1.legend(['ARI'])

        # ax2.plot(list(range(2,maxClust+1)), vmScore, linewidth=2.5)
        # # ax2.set_xticks(list(range(2,maxClust+1)))
        # ax2.set_xlabel('Number of Clusters')
        # ax2.set_ylabel('V-Measure')
        # ax2.set_title('V-Measure')
        # ax2.legend(['V-Measure'])

        # ax3.plot(list(range(2,maxClust+1)), silScoreEuc, linewidth=2.5)
        # ax3.plot(list(range(2,maxClust+1)), silScoreMan, linewidth=2.5)
        # ax3.plot(list(range(2,maxClust+1)), silScoreMah, linewidth=2.5)
        # # ax3.set_xticks(list(range(2,maxClust+1)))
        # ax3.set_xlabel('Number of Clusters')
        # ax3.set_ylabel('Silhouette Coefficient')
        # ax3.set_title('Silhouette Score')
        # ax3.legend(['Euclidean', 'Manhattan', 'Mahalanobis'])

        # plt.suptitle('BNA - Expectation Maximization Clustering Evaluation')
        # plt.show()
        # quit()


        # version with "optimal" 5 clusters
        emClustering = GaussianMixture(n_components=5, covariance_type='full', \
                                       max_iter=100).fit(X_train, y_train)
        emPredictions = emClustering.predict(X_train)

        # assess the clustering performance with Adjusted Rand Index
        ari = metrics.adjusted_rand_score(y_train, emPredictions)

        # reduce dimensionality of data for visualization
        embedding = MDS(n_components=2, verbose=10)
        # X_transformed = embedding.fit_transform(X_train)
        X_transformed = embedding.fit_transform(np.vstack((X_train, emClustering.means_)))
        centroid2D = X_transformed[-5:,:]
        X_transformed = X_transformed[:-5,:]

        #convert predicitons and true labels to colors for visualzation
        cmap = cm.Set1
        cmap2 = cm.tab10
        predictColor = []
        trueColor = []
        for i in range(len(y_train)):
            predictColor.extend([cmap(emPredictions[i])])
            trueColor.extend([cmap2(y_train[i])])


        f = plt.figure()
        plt.scatter(X_transformed[:,0], X_transformed[:,1], c = predictColor, \
                    edgecolors=trueColor)
        for i in range(centroid2D.shape[0]):
            plt.scatter(centroid2D[i,0], centroid2D[i,1], s=400, linewidth=3, \
                        marker = 'X', color = cmap(i), edgecolors='k')
        plt.title('Expectation Maximiation Clustering (5 Clusters)\nARI = ' + str(ari))
        plt.show()


    if fma:
        numClasses = 8

        # create encoder to map from string to int labels
        le = preprocessing.LabelEncoder()
        le.fit(y_train.iloc[:].values)
        intLabels = le.transform(y_train.iloc[:].values)

        emClustering = GaussianMixture(n_components=numClasses, covariance_type='full', \
                                       max_iter=100).fit(X_train, y_train)
        emPredictions = emClustering.predict(X_train)

        # assess the clustering performance with Adjusted Rand Index
        ari = metrics.adjusted_rand_score(y_train, emPredictions)

        # reduce dimensionality of data for visualization
        embedding = MDS(n_components=2, verbose=10, max_iter=1000, n_jobs=-1)
        X_transformed = embedding.fit_transform(np.vstack((X_train, emClustering.means_)))
        centroid2D = X_transformed[-numClasses:,:]
        X_transformed = X_transformed[:-numClasses,:]

        #convert predicitons and true labels to colors for visualzation
        cmap = cm.Set1
        predictColor = []
        trueColor = []
        for i in range(len(y_train)):
            predictColor.extend([cmap(emPredictions[i])])
            trueColor.extend([cmap(intLabels[i])])

        f = plt.figure()
        plt.scatter(X_transformed[:,0], X_transformed[:,1], c = predictColor, \
                    edgecolors=trueColor)
        for i in range(centroid2D.shape[0]):
            plt.scatter(centroid2D[i,0], centroid2D[i,1], s=400, linewidth=3, \
                        marker = 'X', color = cmap(i), edgecolors='k')
        plt.title('Expectation Maximization Clustering\nARI = ' + str(ari))
        plt.show()

# Perform PCA on data

def calcLoss(orig, reco):
    return ((orig - reco) ** 2).mean()

if pComp:
    if bna:

        # randSeeds = np.random.randint(101, size=(3, 5, 4))
        # print(randSeeds)

        pcaRecoErrors = np.zeros(4)
        icaRecoErrors = np.zeros(4)
        srpRecoErrors = np.zeros((10000,4))
        kbRecoErrors = np.zeros(4)

        pcaRecoDur = np.zeros(4)
        icaRecoDur = np.zeros(4)
        srpRecoDur = np.zeros((10000,4))
        kbRecoDur = np.zeros(4)

        for iComps in range(1,len(pcaRecoErrors)+1):

            #PCA
            pca = PCA(n_components=iComps)
            pcaStart = time.time()
            X_train_pca = pca.fit_transform(X_train)
            pcaEnd = time.time()
            X_train_recon = pca.inverse_transform(X_train_pca)
            pcaLoss = calcLoss(X_train, X_train_recon)
            pcaRecoErrors[iComps-1] = pcaLoss
            pcaRecoDur[iComps-1] = pcaEnd - pcaStart

            #ICA
            transformer = FastICA(n_components=iComps, max_iter=500, tol=0.1)
            icaStart = time.time()
            X_transformed = transformer.fit_transform(X_train)
            icaEnd = time.time()
            X_transformed_recon = transformer.inverse_transform(X_transformed)
            icaLoss = calcLoss(X_train, X_transformed_recon)
            icaRecoErrors[iComps-1] = icaLoss
            icaRecoDur[iComps-1] = icaEnd - icaStart

            for iIter in range(10000):
                #Sparce Random Projection
                transformer = random_projection.SparseRandomProjection(n_components=iComps)
                srpStart = time.time()
                X_new = transformer.fit_transform(X_train)
                srpEnd = time.time()
                # X_train_srp_inverse = np.array(X_new). \
                #     dot(transformer.components_) + np.array(X_train.mean(axis=0))
                X_train_srp_inverse = np.array(X_new) * transformer.components_ + np.array(X_train.mean(axis=0))
                srpLoss = calcLoss(X_train, X_train_srp_inverse)
                srpRecoErrors[iIter,iComps-1] = srpLoss
                srpRecoDur[iIter,iComps-1] = srpEnd - srpStart

            #Select k best
            transformer = SelectKBest(mutual_info_classif, k=iComps)
            kbStart = time.time()
            X_new = transformer.fit_transform(X_train, y_train)
            kbEnd = time.time()
            X_train_kBest_recon = transformer.inverse_transform(X_new)
            kBestLoss = calcLoss(X_train, X_train_kBest_recon)
            kbRecoErrors[iComps-1] = kBestLoss
            kbRecoDur[iComps-1] = kbEnd - kbStart


        #Reconstruction Error Fig
        fig, (ax1, ax2) = plt.subplots(1,2)

        ax1.plot(list(range(1,5)), pcaRecoErrors, linewidth=2.5)
        ax1.plot(list(range(1,5)), icaRecoErrors, '--', linewidth=2.5)
        ax1.errorbar(list(range(1,5)), np.mean(srpRecoErrors,0), \
            np.std(srpRecoErrors,0), linewidth=2.5)
        ax1.plot(list(range(1,5)), kbRecoErrors, linewidth=2.5)
        ax1.set_xlabel('Components Used')
        ax1.set_ylabel('Reconstruction Error (MSE)')
        ax1.set_title('BNA - Decomposition Reconstruction Error')
        ax1.legend(['PCA', 'ICA', 'K Best', 'Rand Proj'])

        ax2.plot(list(range(1,5)), pcaRecoDur, linewidth=2.5)
        ax2.plot(list(range(1,5)), icaRecoDur, '--', linewidth=2.5)
        ax2.errorbar(list(range(1,5)), np.mean(srpRecoDur,0), \
            np.std(srpRecoDur,0), linewidth=2.5)
        ax2.plot(list(range(1,5)), kbRecoDur, linewidth=2.5)
        ax2.set_xlabel('Components Used')
        ax2.set_ylabel('Decomposition Time (sec)')
        ax2.set_title('BNA - Decomposition Time')
        ax2.legend(['PCA', 'ICA', 'K Best', 'Rand Proj'])
        plt.show()

        #Reconstruction Error Fig
        # plt.figure()
        # plt.plot(list(range(1,5)), pcaRecoErrors, linewidth=2.5)
        # plt.plot(list(range(1,5)), icaRecoErrors, '--', linewidth=2.5)
        # plt.errorbar(list(range(1,5)), np.mean(srpRecoErrors,0), np.std(srpRecoErrors,0), linewidth=2.5)
        # plt.plot(list(range(1,5)), kbRecoErrors, linewidth=2.5)
        # plt.xticks(list(range(1,5)))
        # plt.xlabel('Components Used')
        # plt.ylabel('Reconstruction Error (MSE)')
        # plt.title('BNA - Decomposition Reconstruction Error')
        # plt.legend(['PCA', 'ICA', 'K Best', 'Rand Proj'])
        # plt.show()

        #PCA Variance Explained Fig
        # plt.figure()
        # plt.plot(pca.explained_variance_ratio_ * 100)
        # # plt.scatter([0, 1, 2, 3], pca.explained_variance_ratio_ * 100)
        # plt.xticks([0, 1, 2, 3])
        # plt.xlabel('Principal Component Index')
        # plt.ylabel('Percent of Variance Explained')
        # plt.title('BNA PCA - Variance Explained')
        # plt.show()

    if fma:

        numComps = 100

        pcaRecoErrors = np.zeros(numComps)
        icaRecoErrors = np.zeros(numComps)
        srpRecoErrors = np.zeros((100,numComps))
        kbRecoErrors = np.zeros(numComps)

        pcaRecoDur = np.zeros(numComps)
        icaRecoDur = np.zeros(numComps)
        srpRecoDur = np.zeros((100,numComps))
        kbRecoDur = np.zeros(numComps)

        for iComps in range(1,len(pcaRecoErrors)+1):
            print('Num Comps = ' + str(iComps) + ' out of ' + str(numComps))

            #PCA
            print('PCA')
            pca = PCA(n_components=iComps)
            pcaStart = time.time()
            X_train_pca = pca.fit_transform(X_train)
            pcaEnd = time.time()
            X_train_recon = pca.inverse_transform(X_train_pca)
            pcaLoss = calcLoss(X_train.values, X_train_recon)
            pcaRecoErrors[iComps-1] = pcaLoss
            pcaRecoDur[iComps-1] = pcaEnd - pcaStart

            #ICA
            print('ICA')
            transformer = FastICA(n_components=iComps, max_iter=200, tol=0.1)
            icaStart = time.time()
            X_transformed = transformer.fit_transform(X_train)
            icaEnd = time.time()
            X_transformed_recon = transformer.inverse_transform(X_transformed)
            icaLoss = calcLoss(X_train.values, X_transformed_recon)
            icaRecoErrors[iComps-1] = icaLoss
            icaRecoDur[iComps-1] = icaEnd - icaStart

            #Sparce Random Projection
            print('SRA')
            for iIter in range(100):
                transformer = random_projection.SparseRandomProjection(n_components=iComps)
                srpStart = time.time()
                X_new = transformer.fit_transform(X_train)
                srpEnd = time.time()
                # X_train_srp_inverse = np.array(X_new). \
                #     dot(transformer.components_) + np.array(X_train.mean(axis=0))
                X_train_srp_inverse = np.array(X_new) * transformer.components_ + np.array(X_train.mean(axis=0))
                srpLoss = calcLoss(X_train.values, X_train_srp_inverse)
                srpRecoErrors[iIter,iComps-1] = srpLoss
                srpRecoDur[iIter,iComps-1] = srpEnd - srpStart

            #Select k best
            print('K Best')
            transformer = SelectKBest(mutual_info_classif, k=iComps)
            kbStart = time.time()
            X_new = transformer.fit_transform(X_train, y_train)
            kbEnd = time.time()
            X_train_kBest_recon = transformer.inverse_transform(X_new)
            kBestLoss = calcLoss(X_train.values, X_train_kBest_recon)
            kbRecoErrors[iComps-1] = kBestLoss
            kbRecoDur[iComps-1] = kbEnd - kbStart



        #Reconstruction Error Fig
        # plt.figure()
        # plt.plot(list(range(1,numComps+1)), pcaRecoErrors, linewidth=2.5)
        # plt.plot(list(range(1,numComps+1)), icaRecoErrors, '--', linewidth=2.5)
        # plt.errorbar(list(range(1,numComps+1)), np.mean(srpRecoErrors,0), \
        #     np.std(srpRecoErrors,0), linewidth=2.5)
        # plt.plot(list(range(1,numComps+1)), kbRecoErrors, linewidth=2.5)
        # plt.xticks(list(range(1,numComps+1)))
        # plt.xlabel('Components Used')
        # plt.ylabel('Reconstruction Error (MSE)')
        # plt.title('FMA - Decomposition Reconstruction Error')
        # plt.legend(['PCA', 'ICA', 'K Best', 'Rand Proj'])
        # plt.show()

        # #Decomposition time fig
        # plt.figure()
        # plt.plot(list(range(1,numComps+1)), pcaRecoDur, linewidth=2.5)
        # plt.plot(list(range(1,numComps+1)), icaRecoDur, '--', linewidth=2.5)
        # plt.errorbar(list(range(1,numComps+1)), np.mean(srpRecoDur,0), \
        #     np.std(srpRecoDur,0), linewidth=2.5)
        # plt.plot(list(range(1,numComps+1)), kbRecoDur, linewidth=2.5)
        # plt.xticks(list(range(1,numComps+1)))
        # plt.xlabel('Components Used')
        # plt.ylabel('Decomposition Time (sec)')
        # plt.title('FMA - Decomposition Time')
        # plt.legend(['PCA', 'ICA', 'K Best', 'Rand Proj'])
        # plt.show()

        fig, (ax1, ax2) = plt.subplots(1,2)

        ax1.plot(list(range(1,numComps+1)), pcaRecoErrors, linewidth=2.5)
        ax1.plot(list(range(1,numComps+1)), icaRecoErrors, '--', linewidth=2.5)
        ax1.errorbar(list(range(1,numComps+1)), np.mean(srpRecoErrors,0), \
            np.std(srpRecoErrors,0), linewidth=2.5)
        ax1.plot(list(range(1,numComps+1)), kbRecoErrors, linewidth=2.5)
        # ax1.set_xticks(list(range(1,numComps+1)))
        ax1.set_xlabel('Components Used')
        ax1.set_ylabel('Reconstruction Error (MSE)')
        ax1.set_title('FMA - Decomposition Reconstruction Error')
        ax1.legend(['PCA', 'ICA', 'K Best', 'Rand Proj'])

        ax2.plot(list(range(1,numComps+1)), pcaRecoDur, linewidth=2.5)
        ax2.plot(list(range(1,numComps+1)), icaRecoDur, '--', linewidth=2.5)
        ax2.errorbar(list(range(1,numComps+1)), np.mean(srpRecoDur,0), \
            np.std(srpRecoDur,0), linewidth=2.5)
        ax2.plot(list(range(1,numComps+1)), kbRecoDur, linewidth=2.5)
        # ax2.set_xticks(list(range(1,numComps+1)))
        ax2.set_xlabel('Components Used')
        ax2.set_ylabel('Decomposition Time (sec)')
        ax2.set_title('FMA - Decomposition Time')
        ax2.legend(['PCA', 'ICA', 'K Best', 'Rand Proj'])
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1,2)

        ax1.plot(list(range(1,numComps+1)), pcaRecoErrors, linewidth=2.5)
        ax1.plot(list(range(1,numComps+1)), icaRecoErrors, '--', linewidth=2.5)
        # ax1.errorbar(list(range(1,numComps+1)), np.mean(srpRecoErrors,0), \
        #     np.std(srpRecoErrors,0), linewidth=2.5)
        ax1.plot(list(range(1,numComps+1)), kbRecoErrors, linewidth=2.5)
        # ax1.set_xticks(list(range(1,numComps+1)))
        ax1.set_xlabel('Components Used')
        ax1.set_ylabel('Reconstruction Error (MSE)')
        ax1.set_title('FMA - Decomposition Reconstruction Error')
        ax1.legend(['PCA', 'ICA', 'K Best'])

        ax2.plot(list(range(1,numComps+1)), pcaRecoDur, linewidth=2.5)
        ax2.plot(list(range(1,numComps+1)), icaRecoDur, '--', linewidth=2.5)
        ax2.errorbar(list(range(1,numComps+1)), np.mean(srpRecoDur,0), \
            np.std(srpRecoDur,0), linewidth=2.5)
        # ax2.plot(list(range(1,numComps+1)), kbRecoDur, linewidth=2.5)
        # ax2.set_xticks(list(range(1,numComps+1)))
        ax2.set_xlabel('Components Used')
        ax2.set_ylabel('Decomposition Time (sec)')
        ax2.set_title('FMA - Decomposition Time')
        ax2.legend(['PCA', 'ICA', 'Rand Proj'])
        plt.show()

if clust2:

    def clusterData(data, trueLabel, nClusters):
        kClustering = KMeans(n_clusters=nClusters, random_state=0).fit(data, trueLabel)

        # assess the clustering performance with Adjusted Rand Index
        ari = metrics.adjusted_rand_score(y_train, kClustering.labels_)

        # reduce dimensionality of data for visualization
        embedding = MDS(n_components=2, verbose=10)
        # print(X_train)
        # print(kClustering.cluster_centers_)
        X_transformed = embedding.fit_transform(np.vstack((data, kClustering.cluster_centers_)))
        centroid2D = X_transformed[-2:,:]
        X_transformed = X_transformed[:-2,:]

        #convert predicitons and true labels to colors for visualzation
        cmap = cm.Set1
        predictColor = []
        trueColor = []
        for i in range(len(y_train)):
            # if kClustering.labels_[i] == 0:
            #     predictColor.extend([cmap(1)])
            # elif kClustering.labels_[i] == 1:
            #     predictColor.extend([cmap(4)])

            # if y_train[i] == 0:
            #     trueColor.extend([cmap(1)])
            # elif y_train[i] == 1:
            #     trueColor.extend([cmap(4)])

            if kClustering.labels_[i] == 0:
                predictColor.extend([cmap(1)])
            elif kClustering.labels_[i] == 1:
                predictColor.extend([cmap(4)])

            if y_train[i] == 0:
                trueColor.extend([cmap(4)])
            elif y_train[i] == 1:
                trueColor.extend([cmap(1)])


        f = plt.figure()
        plt.scatter(X_transformed[:,0], X_transformed[:,1], c = predictColor, \
                    edgecolors=trueColor)
        plt.scatter(centroid2D[0,0], centroid2D[0,1], s=400, linewidth=3, \
                    marker = 'X', color = cmap(1), edgecolors='k')
        plt.scatter(centroid2D[1,0], centroid2D[1,1], s=400, linewidth=3, \
                    marker = 'X', color = cmap(4), edgecolors='k')
        plt.title('Kmeans Clustering\nARI = ' + str(ari))
        plt.show()

        return ari

    if bna:

        numComps = 4

        pcaARI = np.zeros(numComps)
        icaARI = np.zeros(numComps)
        srpARI = np.zeros((100,numComps))
        kbARI = np.zeros(numComps)

        pcaVM = np.zeros(numComps)
        icaVM = np.zeros(numComps)
        srpVM = np.zeros((100,numComps))
        kbVM = np.zeros(numComps)

        for iComps in range(1,len(pcaARI)+1):
            print('Num Comps = ' + str(iComps) + ' out of ' + str(numComps))

            #PCA
            print('PCA')
            pca = PCA(n_components=iComps)
            X_train_pca = pca.fit_transform(X_train)
            kClustering = KMeans(n_clusters=2, random_state=0).fit(X_train_pca, y_train)
            pcaARI[iComps-1] = metrics.adjusted_rand_score(y_train, kClustering.labels_)
            pcaVM[iComps-1] = metrics.v_measure_score(y_train, kClustering.labels_)

            #ICA
            print('ICA')
            transformer = FastICA(n_components=iComps, max_iter=200, tol=0.1)
            X_transformed = transformer.fit_transform(X_train)
            kClustering = KMeans(n_clusters=2, random_state=0).fit(X_transformed, y_train)
            icaARI[iComps-1] = metrics.adjusted_rand_score(y_train, kClustering.labels_)
            icaVM[iComps-1] = metrics.v_measure_score(y_train, kClustering.labels_)

            #Sparce Random Projection
            print('SRA')
            for iIter in range(100):
                transformer = random_projection.SparseRandomProjection(n_components=iComps)
                X_new = transformer.fit_transform(X_train)
                kClustering = KMeans(n_clusters=2, random_state=0).fit(X_new, y_train)
                srpARI[iIter,iComps-1] = metrics.adjusted_rand_score(y_train, kClustering.labels_)
                srpVM[iIter,iComps-1] = metrics.v_measure_score(y_train, kClustering.labels_)

            #Select k best
            print('K Best')
            transformer = SelectKBest(mutual_info_classif, k=iComps)
            X_new = transformer.fit_transform(X_train, y_train)
            kClustering = KMeans(n_clusters=2, random_state=0).fit(X_new, y_train)
            kbARI[iComps-1] = metrics.adjusted_rand_score(y_train, kClustering.labels_)
            kbVM[iComps-1] = metrics.v_measure_score(y_train, kClustering.labels_)

        fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)

        ax1.plot(list(range(1,numComps+1)), pcaARI, linewidth=2.5)
        ax1.plot(list(range(1,numComps+1)), icaARI, '--', linewidth=2.5)
        ax1.errorbar(list(range(1,numComps+1)), np.mean(srpARI,0), \
            np.std(srpARI,0), linewidth=2.5)
        ax1.plot(list(range(1,numComps+1)), kbARI, linewidth=2.5)
        # ax1.set_xticks(list(range(1,numComps+1)))
        ax1.set_xlabel('Components Used')
        ax1.set_ylabel('Adjusted Rand Index')
        ax1.set_title('Adjusted Rand Index')
        ax1.legend(['PCA', 'ICA', 'K Best', 'Rand Proj'])

        ax2.plot(list(range(1,numComps+1)), pcaVM, linewidth=2.5)
        ax2.plot(list(range(1,numComps+1)), icaVM, '--', linewidth=2.5)
        ax2.errorbar(list(range(1,numComps+1)), np.mean(srpVM,0), \
            np.std(srpVM,0), linewidth=2.5)
        ax2.plot(list(range(1,numComps+1)), kbVM, linewidth=2.5)
        # ax2.set_xticks(list(range(1,numComps+1)))
        ax2.set_xlabel('Components Used')
        ax2.set_ylabel('V-Measure')
        ax2.set_title('V-Measure')
        ax2.legend(['PCA', 'ICA', 'K Best', 'Rand Proj'])
        plt.suptitle('BNA - K-Means Clustering Evaluation after Dimensionality Reduction')
        plt.show()

        # f, ax = plt.subplots(4,3,sharey=True, sharex=True)
        f, ax = plt.subplots(4,3, gridspec_kw = {'wspace':0.025, 'hspace':0.05})

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        ariVals = np.zeros((4,3))

        rowIdx = 0
        for row in ax:
            colIdx = 0
            for col in row:

                if rowIdx==0:
                    # if colIdx==0:
                    pca = PCA(n_components=colIdx+2)
                    X_train_pca = pca.fit_transform(X_train)
                    kClustering = KMeans(n_clusters=2, random_state=0).fit(X_train_pca, y_train)

                    # assess the clustering performance with Adjusted Rand Index
                    ariVals[rowIdx, colIdx] = metrics.adjusted_rand_score(y_train, kClustering.labels_)

                    # reduce dimensionality of data for visualization
                    embedding = MDS(n_components=2, verbose=10)
                    X_transformed = embedding.fit_transform(np.vstack((X_train_pca, kClustering.cluster_centers_)))
                    centroid2D = X_transformed[-2:,:]
                    X_transformed = X_transformed[:-2,:]

                    #convert predicitons and true labels to colors for visualzation
                    cmap = cm.Set1
                    predictColor = []
                    trueColor = []
                    for i in range(len(y_train)):
                        if kClustering.labels_[i] == 0:
                            predictColor.extend([cmap(1)])
                        elif kClustering.labels_[i] == 1:
                            predictColor.extend([cmap(4)])

                        if y_train[i] == 0:
                            trueColor.extend([cmap(4)])
                        elif y_train[i] == 1:
                            trueColor.extend([cmap(1)])


                    col.scatter(X_transformed[:,0], X_transformed[:,1], s=10, linewidth=0.5, c = predictColor, \
                                edgecolors=trueColor)
                    col.scatter(centroid2D[0,0], centroid2D[0,1], s=200, linewidth=3, \
                                marker = 'X', color = cmap(1), edgecolors='k')
                    col.scatter(centroid2D[1,0], centroid2D[1,1], s=200, linewidth=3, \
                                marker = 'X', color = cmap(4), edgecolors='k')
                    # col.set_title('Kmeans Clustering\nARI = ' + str(ari))
                    col.set_xticks([])
                    col.set_yticks([])
                    # col.set_title('ARI = ' + str(ariVals[rowIdx, colIdx]))

                    col.text(0.05, 0.95, 'ARI = ' + str(round(ariVals[rowIdx, colIdx],4)),
                        transform=col.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)

                    if colIdx == 0:
                        col.set_ylabel('PCA',rotation=0, fontsize=18, labelpad=50)
                        col.set_title('2 Components/Features', fontsize=18)
                    if colIdx == 1:
                        col.set_title('3 Components/Features', fontsize=18)
                    if colIdx == 2:
                        col.set_title('4 Components/Features', fontsize=18)
                    # plt.show()
                    # quit()

                if rowIdx==1:
                    transformer = FastICA(n_components=colIdx+2, max_iter=500, tol=0.1)
                    X_transformed = transformer.fit_transform(X_train)
                    kClustering = KMeans(n_clusters=2, random_state=0).fit(X_transformed, y_train)

                    # assess the clustering performance with Adjusted Rand Index
                    ariVals[rowIdx, colIdx] = metrics.adjusted_rand_score(y_train, kClustering.labels_)

                    # reduce dimensionality of data for visualization
                    embedding = MDS(n_components=2, verbose=10)
                    X_transformed = embedding.fit_transform(np.vstack((X_transformed, kClustering.cluster_centers_)))
                    centroid2D = X_transformed[-2:,:]
                    X_transformed = X_transformed[:-2,:]

                    #convert predicitons and true labels to colors for visualzation
                    cmap = cm.Set1
                    predictColor = []
                    trueColor = []
                    for i in range(len(y_train)):
                        if kClustering.labels_[i] == 0:
                            predictColor.extend([cmap(1)])
                        elif kClustering.labels_[i] == 1:
                            predictColor.extend([cmap(4)])

                        if y_train[i] == 0:
                            trueColor.extend([cmap(4)])
                        elif y_train[i] == 1:
                            trueColor.extend([cmap(1)])


                    col.scatter(X_transformed[:,0], X_transformed[:,1], s=10, linewidth=0.5, c = predictColor, \
                                edgecolors=trueColor)
                    col.scatter(centroid2D[0,0], centroid2D[0,1], s=200, linewidth=3, \
                                marker = 'X', color = cmap(1), edgecolors='k')
                    col.scatter(centroid2D[1,0], centroid2D[1,1], s=200, linewidth=3, \
                                marker = 'X', color = cmap(4), edgecolors='k')
                    # col.set_title('Kmeans Clustering\nARI = ' + str(ari))
                    col.set_xticks([])
                    col.set_yticks([])
                    # col.set_title('ARI = ' + str(ariVals[rowIdx, colIdx]))
                    col.text(0.05, 0.95, 'ARI = ' + str(round(ariVals[rowIdx, colIdx],4)),
                        transform=col.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)

                    if colIdx == 0:
                        col.set_ylabel('ICA',rotation=0, fontsize=18, labelpad=50)

                if rowIdx==2:
                    transformer = random_projection.SparseRandomProjection(n_components=colIdx+2)
                    X_new = transformer.fit_transform(X_train)
                    kClustering = KMeans(n_clusters=2, random_state=0).fit(X_new, y_train)

                    # assess the clustering performance with Adjusted Rand Index
                    ariVals[rowIdx, colIdx] = metrics.adjusted_rand_score(y_train, kClustering.labels_)

                    # reduce dimensionality of data for visualization
                    embedding = MDS(n_components=2, verbose=10)
                    X_transformed = embedding.fit_transform(np.vstack((X_new, kClustering.cluster_centers_)))
                    centroid2D = X_transformed[-2:,:]
                    X_transformed = X_transformed[:-2,:]

                    #convert predicitons and true labels to colors for visualzation
                    cmap = cm.Set1
                    predictColor = []
                    trueColor = []
                    for i in range(len(y_train)):
                        if kClustering.labels_[i] == 0:
                            predictColor.extend([cmap(1)])
                        elif kClustering.labels_[i] == 1:
                            predictColor.extend([cmap(4)])

                        if y_train[i] == 0:
                            trueColor.extend([cmap(4)])
                        elif y_train[i] == 1:
                            trueColor.extend([cmap(1)])


                    col.scatter(X_transformed[:,0], X_transformed[:,1], s=10, linewidth=0.5, c = predictColor, \
                                edgecolors=trueColor)
                    col.scatter(centroid2D[0,0], centroid2D[0,1], s=200, linewidth=3, \
                                marker = 'X', color = cmap(1), edgecolors='k')
                    col.scatter(centroid2D[1,0], centroid2D[1,1], s=200, linewidth=3, \
                                marker = 'X', color = cmap(4), edgecolors='k')
                    # col.set_title('Kmeans Clustering\nARI = ' + str(ari))
                    col.set_xticks([])
                    col.set_yticks([])
                    # col.set_title('ARI = ' + str(ariVals[rowIdx, colIdx]))
                    col.text(0.05, 0.95, 'ARI = ' + str(round(ariVals[rowIdx, colIdx],4)),
                        transform=col.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)

                    if colIdx == 0:
                        col.set_ylabel('SRP',rotation=0, fontsize=18, labelpad=50)

                if rowIdx==3:
                    transformer = SelectKBest(mutual_info_classif, k=colIdx+2)
                    X_new = transformer.fit_transform(X_train, y_train)
                    kClustering = KMeans(n_clusters=2, random_state=0).fit(X_new, y_train)

                    # assess the clustering performance with Adjusted Rand Index
                    ariVals[rowIdx, colIdx] = metrics.adjusted_rand_score(y_train, kClustering.labels_)

                    # reduce dimensionality of data for visualization
                    embedding = MDS(n_components=2, verbose=10)
                    X_transformed = embedding.fit_transform(np.vstack((X_new, kClustering.cluster_centers_)))
                    centroid2D = X_transformed[-2:,:]
                    X_transformed = X_transformed[:-2,:]

                    #convert predicitons and true labels to colors for visualzation
                    cmap = cm.Set1
                    predictColor = []
                    trueColor = []
                    for i in range(len(y_train)):
                        if kClustering.labels_[i] == 0:
                            predictColor.extend([cmap(1)])
                        elif kClustering.labels_[i] == 1:
                            predictColor.extend([cmap(4)])

                        if y_train[i] == 0:
                            trueColor.extend([cmap(4)])
                        elif y_train[i] == 1:
                            trueColor.extend([cmap(1)])


                    col.scatter(X_transformed[:,0], X_transformed[:,1], s=10, linewidth=0.5, c = predictColor, \
                                edgecolors=trueColor)
                    col.scatter(centroid2D[0,0], centroid2D[0,1], s=200, linewidth=3, \
                                marker = 'X', color = cmap(1), edgecolors='k')
                    col.scatter(centroid2D[1,0], centroid2D[1,1], s=200, linewidth=3, \
                                marker = 'X', color = cmap(4), edgecolors='k')
                    # col.set_title('Kmeans Clustering\nARI = ' + str(ari))
                    col.set_xticks([])
                    col.set_yticks([])
                    # col.set_title('ARI = ' + str(ariVals[rowIdx, colIdx]))
                    col.text(0.05, 0.95, 'ARI = ' + str(round(ariVals[rowIdx, colIdx],4)),
                        transform=col.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)

                    if colIdx == 0:
                        col.set_ylabel('KBest',rotation=0, fontsize=18, labelpad=50)

                colIdx += 1
            rowIdx += 1

        # print(ariVals)
        plt.show()
        # quit()


        # for iComps in range(2,5):

        #     iComps = 4

        #     #PCA
        #     pca = PCA(n_components=iComps)
        #     X_train_pca = pca.fit_transform(X_train)
        #     pcaARI = clusterData(X_train_pca, y_train, 2)
        #     # print(X_train_pca)
        #     quit()

        #     #ICA
        #     transformer = FastICA(n_components=iComps, max_iter=500, tol=0.1)
        #     X_transformed = transformer.fit_transform(X_train)
        #     # icaARI = clusterData(X_transformed, y_train, 2)
        #     # quit()

        #     #Sparce Random Projection
        #     transformer = random_projection.SparseRandomProjection(n_components=iComps)
        #     X_new = transformer.fit_transform(X_train)

        #     #Select k best
        #     transformer = SelectKBest(mutual_info_classif, k=iComps)
        #     X_new = transformer.fit_transform(X_train, y_train)
        #     kbestARI = clusterData(X_new, y_train, 2)
        #     quit()

    if fma:

        le = preprocessing.LabelEncoder()
        le.fit(y_train.iloc[:].values)
        intLabels = le.transform(y_train.iloc[:].values)

        numComps = 100

        pcaARI = np.zeros(numComps)
        icaARI = np.zeros(numComps)
        srpARI = np.zeros((100,numComps))
        kbARI = np.zeros(numComps)

        pcaVM = np.zeros(numComps)
        icaVM = np.zeros(numComps)
        srpVM = np.zeros((100,numComps))
        kbVM = np.zeros(numComps)

        for iComps in range(1,len(pcaARI)+1):
            print('Num Comps = ' + str(iComps) + ' out of ' + str(numComps))

            #PCA
            print('PCA')
            pca = PCA(n_components=iComps)
            X_train_pca = pca.fit_transform(X_train)
            kClustering = KMeans(n_clusters=8, random_state=0).fit(X_train_pca, intLabels)
            pcaARI[iComps-1] = metrics.adjusted_rand_score(intLabels, kClustering.labels_)
            pcaVM[iComps-1] = metrics.v_measure_score(intLabels, kClustering.labels_)

            #ICA
            print('ICA')
            transformer = FastICA(n_components=iComps, max_iter=200, tol=0.1)
            X_transformed = transformer.fit_transform(X_train)
            kClustering = KMeans(n_clusters=8, random_state=0).fit(X_transformed, intLabels)
            icaARI[iComps-1] = metrics.adjusted_rand_score(intLabels, kClustering.labels_)
            icaVM[iComps-1] = metrics.v_measure_score(intLabels, kClustering.labels_)

            #Sparce Random Projection
            print('SRA')
            for iIter in range(100):
                transformer = random_projection.SparseRandomProjection(n_components=iComps)
                X_new = transformer.fit_transform(X_train)
                kClustering = KMeans(n_clusters=8, random_state=0).fit(X_new, intLabels)
                srpARI[iIter,iComps-1] = metrics.adjusted_rand_score(intLabels, kClustering.labels_)
                srpVM[iIter,iComps-1] = metrics.v_measure_score(intLabels, kClustering.labels_)

            #Select k best
            print('K Best')
            transformer = SelectKBest(mutual_info_classif, k=iComps)
            X_new = transformer.fit_transform(X_train, y_train)
            kClustering = KMeans(n_clusters=8, random_state=0).fit(X_new, intLabels)
            kbARI[iComps-1] = metrics.adjusted_rand_score(intLabels, kClustering.labels_)
            kbVM[iComps-1] = metrics.v_measure_score(intLabels, kClustering.labels_)

        fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)

        ax1.plot(list(range(1,numComps+1)), pcaARI, linewidth=2.5)
        ax1.plot(list(range(1,numComps+1)), icaARI, '--', linewidth=2.5)
        ax1.errorbar(list(range(1,numComps+1)), np.mean(srpARI,0), \
            np.std(srpARI,0), linewidth=2.5)
        ax1.plot(list(range(1,numComps+1)), kbARI, linewidth=2.5)
        # ax1.set_xticks(list(range(1,numComps+1)))
        ax1.set_xlabel('Components Used')
        ax1.set_ylabel('Adjusted Rand Index')
        ax1.set_title('Adjusted Rand Index')
        ax1.legend(['PCA', 'ICA', 'K Best', 'Rand Proj'])

        ax2.plot(list(range(1,numComps+1)), pcaVM, linewidth=2.5)
        ax2.plot(list(range(1,numComps+1)), icaVM, '--', linewidth=2.5)
        ax2.errorbar(list(range(1,numComps+1)), np.mean(srpVM,0), \
            np.std(srpVM,0), linewidth=2.5)
        ax2.plot(list(range(1,numComps+1)), kbVM, linewidth=2.5)
        # ax2.set_xticks(list(range(1,numComps+1)))
        ax2.set_xlabel('Components Used')
        ax2.set_ylabel('V-Measure')
        ax2.set_title('V-Measure')
        ax2.legend(['PCA', 'ICA', 'K Best', 'Rand Proj'])
        plt.suptitle('FMA - K-Means Clustering Evaluation after Dimensionality Reduction')
        plt.show()

        # f, ax = plt.subplots(4,3,sharey=True, sharex=True)
        # f, ax = plt.subplots(4,3, gridspec_kw = {'wspace':0.025, 'hspace':0.05})

        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # ariVals = np.zeros((4,3))

        # le = preprocessing.LabelEncoder()
        # le.fit(y_train.iloc[:].values)
        # intLabels = le.transform(y_train.iloc[:].values)

        # rowIdx = 0
        # for row in ax:
        #     colIdx = 0
        #     for col in row:

        #         if colIdx==0:
        #             numComps = 20
        #         elif colIdx==1:
        #             numComps = 40
        #         elif colIdx==2:
        #             numComps = 80

        #         if rowIdx==0:

        #             pca = PCA(n_components=numComps)
        #             X_train_pca = pca.fit_transform(X_train)
        #             kClustering = KMeans(n_clusters=8, random_state=0).fit(X_train_pca, intLabels)

        #             # assess the clustering performance with Adjusted Rand Index
        #             ariVals[rowIdx, colIdx] = metrics.adjusted_rand_score(intLabels, kClustering.labels_)

        #             # reduce dimensionality of data for visualization
        #             embedding = MDS(n_components=2, verbose=10, max_iter=5, n_jobs=-1)
        #             X_transformed = embedding.fit_transform(np.vstack((X_train_pca, kClustering.cluster_centers_)))
        #             centroid2D = X_transformed[-8:,:]
        #             X_transformed = X_transformed[:-8,:]

        #             #convert predicitons and true labels to colors for visualzation
        #             cmap = cm.Set1
        #             predictColor = []
        #             trueColor = []
        #             for i in range(len(y_train)):
        #                 predictColor.extend([cmap(kClustering.labels_[i])])
        #                 trueColor.extend([cmap(intLabels[i])])

        #             col.scatter(X_transformed[:,0], X_transformed[:,1], s=10, linewidth=0.5,
        #                 c = predictColor, edgecolors=trueColor)
        #             for i in range(centroid2D.shape[0]):
        #                 col.scatter(centroid2D[i,0], centroid2D[i,1], s=200, linewidth=3, \
        #                             marker = 'X', color = cmap(i), edgecolors='k')

        #             col.set_xticks([])
        #             col.set_yticks([])
        #             # col.set_title('ARI = ' + str(ariVals[rowIdx, colIdx]))

        #             col.text(0.05, 0.95, 'ARI = ' + str(round(ariVals[rowIdx, colIdx],4)),
        #                 transform=col.transAxes, fontsize=12,
        #                 verticalalignment='top', bbox=props)

        #             if colIdx == 0:
        #                 col.set_ylabel('PCA',rotation=0, fontsize=18, labelpad=50)
        #                 col.set_title('20 Components/Features', fontsize=18)
        #             if colIdx == 1:
        #                 col.set_title('40 Components/Features', fontsize=18)
        #             if colIdx == 2:
        #                 col.set_title('80 Components/Features', fontsize=18)

        #         if rowIdx==1:
        #             transformer = FastICA(n_components=numComps, max_iter=500, tol=0.1)
        #             X_transformed = transformer.fit_transform(X_train)
        #             kClustering = KMeans(n_clusters=8, random_state=0).fit(X_transformed, intLabels)

        #             # assess the clustering performance with Adjusted Rand Index
        #             ariVals[rowIdx, colIdx] = metrics.adjusted_rand_score(intLabels, kClustering.labels_)

        #             # reduce dimensionality of data for visualization
        #             embedding = MDS(n_components=2, verbose=10, max_iter=5, n_jobs=-1)
        #             X_transformed = embedding.fit_transform(np.vstack((X_transformed, kClustering.cluster_centers_)))
        #             centroid2D = X_transformed[-8:,:]
        #             X_transformed = X_transformed[:-8,:]

        #             #convert predicitons and true labels to colors for visualzation
        #             cmap = cm.Set1
        #             predictColor = []
        #             trueColor = []
        #             for i in range(len(y_train)):
        #                 predictColor.extend([cmap(kClustering.labels_[i])])
        #                 trueColor.extend([cmap(intLabels[i])])

        #             col.scatter(X_transformed[:,0], X_transformed[:,1], s=10, linewidth=0.5,
        #                 c = predictColor, edgecolors=trueColor)
        #             for i in range(centroid2D.shape[0]):
        #                 col.scatter(centroid2D[i,0], centroid2D[i,1], s=200, linewidth=3, \
        #                             marker = 'X', color = cmap(i), edgecolors='k')

        #             col.set_xticks([])
        #             col.set_yticks([])
        #             # col.set_title('ARI = ' + str(ariVals[rowIdx, colIdx]))

        #             col.text(0.05, 0.95, 'ARI = ' + str(round(ariVals[rowIdx, colIdx],4)),
        #                 transform=col.transAxes, fontsize=12,
        #                 verticalalignment='top', bbox=props)

        #             if colIdx == 0:
        #                 col.set_ylabel('ICA',rotation=0, fontsize=18, labelpad=50)

        #         if rowIdx==2:
        #             transformer = random_projection.SparseRandomProjection(n_components=numComps)
        #             X_new = transformer.fit_transform(X_train)
        #             kClustering = KMeans(n_clusters=8, random_state=0).fit(X_new, intLabels)

        #             # assess the clustering performance with Adjusted Rand Index
        #             ariVals[rowIdx, colIdx] = metrics.adjusted_rand_score(intLabels, kClustering.labels_)

        #             # reduce dimensionality of data for visualization
        #             embedding = MDS(n_components=2, verbose=10, max_iter=5, n_jobs=-1)
        #             X_transformed = embedding.fit_transform(np.vstack((X_new, kClustering.cluster_centers_)))
        #             centroid2D = X_transformed[-8:,:]
        #             X_transformed = X_transformed[:-8,:]

        #             #convert predicitons and true labels to colors for visualzation
        #             cmap = cm.Set1
        #             predictColor = []
        #             trueColor = []
        #             for i in range(len(y_train)):
        #                 predictColor.extend([cmap(kClustering.labels_[i])])
        #                 trueColor.extend([cmap(intLabels[i])])

        #             col.scatter(X_transformed[:,0], X_transformed[:,1], s=10, linewidth=0.5,
        #                 c = predictColor, edgecolors=trueColor)
        #             for i in range(centroid2D.shape[0]):
        #                 col.scatter(centroid2D[i,0], centroid2D[i,1], s=200, linewidth=3, \
        #                             marker = 'X', color = cmap(i), edgecolors='k')

        #             col.set_xticks([])
        #             col.set_yticks([])
        #             # col.set_title('ARI = ' + str(ariVals[rowIdx, colIdx]))

        #             col.text(0.05, 0.95, 'ARI = ' + str(round(ariVals[rowIdx, colIdx],4)),
        #                 transform=col.transAxes, fontsize=12,
        #                 verticalalignment='top', bbox=props)

        #             if colIdx == 0:
        #                 col.set_ylabel('SRP',rotation=0, fontsize=18, labelpad=50)

        #         if rowIdx==3:
        #             transformer = SelectKBest(mutual_info_classif, k=numComps)
        #             X_new = transformer.fit_transform(X_train, y_train)
        #             kClustering = KMeans(n_clusters=8, random_state=0).fit(X_new, intLabels)

        #             # assess the clustering performance with Adjusted Rand Index
        #             ariVals[rowIdx, colIdx] = metrics.adjusted_rand_score(intLabels, kClustering.labels_)

        #             # reduce dimensionality of data for visualization
        #             embedding = MDS(n_components=2, verbose=10, max_iter=5, n_jobs=-1)
        #             X_transformed = embedding.fit_transform(np.vstack((X_new, kClustering.cluster_centers_)))
        #             centroid2D = X_transformed[-8:,:]
        #             X_transformed = X_transformed[:-8,:]

        #             #convert predicitons and true labels to colors for visualzation
        #             cmap = cm.Set1
        #             predictColor = []
        #             trueColor = []
        #             for i in range(len(y_train)):
        #                 predictColor.extend([cmap(kClustering.labels_[i])])
        #                 trueColor.extend([cmap(intLabels[i])])

        #             col.scatter(X_transformed[:,0], X_transformed[:,1], s=10, linewidth=0.5,
        #                 c = predictColor, edgecolors=trueColor)
        #             for i in range(centroid2D.shape[0]):
        #                 col.scatter(centroid2D[i,0], centroid2D[i,1], s=200, linewidth=3, \
        #                             marker = 'X', color = cmap(i), edgecolors='k')

        #             col.set_xticks([])
        #             col.set_yticks([])
        #             # col.set_title('ARI = ' + str(ariVals[rowIdx, colIdx]))

        #             col.text(0.05, 0.95, 'ARI = ' + str(round(ariVals[rowIdx, colIdx],4)),
        #                 transform=col.transAxes, fontsize=12,
        #                 verticalalignment='top', bbox=props)

        #             if colIdx == 0:
        #                 col.set_ylabel('KBest',rotation=0, fontsize=18, labelpad=50)

        #         colIdx += 1
        #     rowIdx += 1

        # # print(ariVals)
        # plt.show()

if part4:

    # paramSearch = False
    # Training settings
    batch_size = 64

    if fma:
        # create encoder to map from string to int labels
        le = preprocessing.LabelEncoder()
        le.fit(y_train.iloc[:].values)

        #Standard
        # torchTrainX = torch.tensor(X_train.iloc[:,:].values)
        # torchTrainY = torch.tensor(le.transform(y_train.iloc[:].values))

        # torchTestX = torch.tensor(X_test.iloc[:,:].values)
        # torchTestY = torch.tensor(le.transform(y_test.iloc[:].values))

        # torchValX = torch.tensor(X_val.iloc[:,:].values)
        # torchValY = torch.tensor(le.transform(y_val.iloc[:].values))

        #PCA
        # if not part5:
        #     numFeats = 500
        #     pca = PCA(n_components=numFeats)
        #     X_train_pca = pca.fit_transform(X_train)
        #     X_val_pca = pca.transform(X_val)
        #     X_test_pca = pca.transform(X_test)

        #     # loaders
        #     torchTrainX = torch.tensor(X_train_pca)
        #     torchTrainY = torch.tensor(le.transform(y_train.iloc[:].values))

        #     torchTestX = torch.tensor(X_test_pca)
        #     torchTestY = torch.tensor(le.transform(y_test.iloc[:].values))

        #     torchValX = torch.tensor(X_val_pca)
        #     torchValY = torch.tensor(le.transform(y_val.iloc[:].values))
        # elif part5:
        #     numFeats = 500
        #     pca = PCA(n_components=numFeats-8)
        #     X_train_pca = pca.fit_transform(X_train)
        #     kmeans = KMeans(n_clusters=8, random_state=0).fit(X_train_pca, y_train)
        #     kClustering = kmeans.predict(X_train_pca)
        #     kClustering = OneHotEncoder().fit_transform(np.expand_dims(kClustering, axis=1)).toarray()
        #     X_train_pca_aug = np.hstack((kClustering,X_train_pca))

        #     # print(kClustering[-1,:])
        #     # quit()

        #     X_val_pca = pca.transform(X_val)
        #     X_val_clust = kmeans.transform(X_val_pca)
        #     X_val_pca_aug = np.hstack((X_val_clust,X_val_pca))

        #     X_test_pca = pca.transform(X_test)
        #     X_test_clust = kmeans.transform(X_test_pca)
        #     X_test_pca_aug = np.hstack((X_test_clust,X_test_pca))

        #     # loaders
        #     torchTrainX = torch.tensor(X_train_pca_aug)
        #     torchTrainY = torch.tensor(le.transform(y_train.iloc[:].values))

        #     torchTestX = torch.tensor(X_test_pca_aug)
        #     torchTestY = torch.tensor(le.transform(y_test.iloc[:].values))

        #     torchValX = torch.tensor(X_val_pca_aug)
        #     torchValY = torch.tensor(le.transform(y_val.iloc[:].values))

        #ICA
        # if not part5:
        #     numFeats = 500
        #     transformer = FastICA(n_components=numFeats)
        #     X_train_ica = transformer.fit_transform(X_train)
        #     X_val_ica = transformer.transform(X_val)
        #     X_test_ica = transformer.transform(X_test)

        #     torchTrainX = torch.tensor(X_train_ica)
        #     torchTrainY = torch.tensor(le.transform(y_train.iloc[:].values))

        #     torchTestX = torch.tensor(X_test_ica)
        #     torchTestY = torch.tensor(le.transform(y_test.iloc[:].values))

        #     torchValX = torch.tensor(X_val_ica)
        #     torchValY = torch.tensor(le.transform(y_val.iloc[:].values))
        # elif part5:
        #     numFeats = 500
        #     transformer = FastICA(n_components=numFeats-8)
        #     X_train_ica = transformer.fit_transform(X_train)
        #     kmeans = KMeans(n_clusters=8, random_state=0).fit(X_train_ica, y_train)
        #     kClustering = kmeans.predict(X_train_ica)
        #     kClustering = OneHotEncoder().fit_transform(np.expand_dims(kClustering, axis=1)).toarray()
        #     X_train_ica_aug = np.hstack((kClustering,X_train_ica))

        #     X_val_ica = transformer.transform(X_val)
        #     X_val_clust = kmeans.transform(X_val_ica)
        #     X_val_ica_aug = np.hstack((X_val_clust,X_val_ica))

        #     X_test_ica = transformer.transform(X_test)
        #     X_test_clust = kmeans.transform(X_test_ica)
        #     X_test_ica_aug = np.hstack((X_test_clust,X_test_ica))

        #     # loaders
        #     torchTrainX = torch.tensor(X_train_ica_aug)
        #     torchTrainY = torch.tensor(le.transform(y_train.iloc[:].values))

        #     torchTestX = torch.tensor(X_test_ica_aug)
        #     torchTestY = torch.tensor(le.transform(y_test.iloc[:].values))

        #     torchValX = torch.tensor(X_val_ica_aug)
        #     torchValY = torch.tensor(le.transform(y_val.iloc[:].values))

        #SRP
        # if not part5:
        #     numFeats = 500
        #     transformer = random_projection.SparseRandomProjection(n_components=numFeats)
        #     X_train_srp = transformer.fit_transform(X_train)
        #     X_val_srp = transformer.transform(X_val)
        #     X_test_srp = transformer.transform(X_test)

        #     torchTrainX = torch.tensor(X_train_srp)
        #     torchTrainY = torch.tensor(le.transform(y_train.iloc[:].values))

        #     torchTestX = torch.tensor(X_test_srp)
        #     torchTestY = torch.tensor(le.transform(y_test.iloc[:].values))

        #     torchValX = torch.tensor(X_val_srp)
        #     torchValY = torch.tensor(le.transform(y_val.iloc[:].values))
        # elif part5:
        #     numFeats = 500
        #     transformer = random_projection.SparseRandomProjection(n_components=numFeats-8)
        #     X_train_srp = transformer.fit_transform(X_train)
        #     kmeans = KMeans(n_clusters=8, random_state=0).fit(X_train_srp, y_train)
        #     kClustering = kmeans.predict(X_train_srp)
        #     kClustering = OneHotEncoder().fit_transform(np.expand_dims(kClustering, axis=1)).toarray()
        #     X_train_srp_aug = np.hstack((kClustering,X_train_srp))

        #     X_val_srp = transformer.transform(X_val)
        #     X_val_clust = kmeans.transform(X_val_srp)
        #     X_val_srp_aug = np.hstack((X_val_clust,X_val_srp))

        #     X_test_srp = transformer.transform(X_test)
        #     X_test_clust = kmeans.transform(X_test_srp)
        #     X_test_srp_aug = np.hstack((X_test_clust,X_test_srp))

        #     # loaders
        #     torchTrainX = torch.tensor(X_train_srp_aug)
        #     torchTrainY = torch.tensor(le.transform(y_train.iloc[:].values))

        #     torchTestX = torch.tensor(X_test_srp_aug)
        #     torchTestY = torch.tensor(le.transform(y_test.iloc[:].values))

        #     torchValX = torch.tensor(X_val_srp_aug)
        #     torchValY = torch.tensor(le.transform(y_val.iloc[:].values))

        #KBest
        if not part5:
            numFeats = 500
            transformer = SelectKBest(mutual_info_classif, k=numFeats)
            X_train_kb = transformer.fit_transform(X_train, y_train)
            X_val_kb = transformer.transform(X_val)
            X_test_kb = transformer.transform(X_test)

            torchTrainX = torch.tensor(X_train_kb)
            torchTrainY = torch.tensor(le.transform(y_train.iloc[:].values))

            torchTestX = torch.tensor(X_test_kb)
            torchTestY = torch.tensor(le.transform(y_test.iloc[:].values))

            torchValX = torch.tensor(X_val_kb)
            torchValY = torch.tensor(le.transform(y_val.iloc[:].values))

        elif part5:
            numFeats = 500
            transformer = SelectKBest(mutual_info_classif, k=numFeats-8)
            X_train_kb = transformer.fit_transform(X_train, y_train)
            kmeans = KMeans(n_clusters=8, random_state=0).fit(X_train_kb, y_train)
            kClustering = kmeans.predict(X_train_kb)
            kClustering = OneHotEncoder().fit_transform(np.expand_dims(kClustering, axis=1)).toarray()
            X_train_kb_aug = np.hstack((kClustering,X_train_kb))

            X_val_kb = transformer.transform(X_val)
            X_val_clust = kmeans.transform(X_val_kb)
            X_val_kb_aug = np.hstack((X_val_clust,X_val_kb))

            X_test_kb = transformer.transform(X_test)
            X_test_clust = kmeans.transform(X_test_kb)
            X_test_kb_aug = np.hstack((X_test_clust,X_test_kb))

            # loaders
            torchTrainX = torch.tensor(X_train_kb_aug)
            torchTrainY = torch.tensor(le.transform(y_train.iloc[:].values))

            torchTestX = torch.tensor(X_test_kb_aug)
            torchTestY = torch.tensor(le.transform(y_test.iloc[:].values))

            torchValX = torch.tensor(X_val_kb_aug)
            torchValY = torch.tensor(le.transform(y_val.iloc[:].values))

    elif bna:

        #standard
        torchTrainX = torch.tensor(X_train)
        torchTrainY = torch.tensor(y_train, dtype=torch.long)

        torchValX = torch.tensor(X_val)
        torchValY = torch.tensor(y_val, dtype=torch.long)

        torchTestX = torch.tensor(X_test)
        torchTestY = torch.tensor(y_test, dtype=torch.long)


        #PCA
        # pca = PCA(n_components=4)
        # X_train_pca = pca.fit_transform(X_train)
        # X_val_pca = pca.transform(X_val)
        # X_test_pca = pca.transform(X_test)

        # torchTrainX = torch.tensor(X_train_pca)
        # torchTrainY = torch.tensor(y_train, dtype=torch.long)

        # torchValX = torch.tensor(X_val_pca)
        # torchValY = torch.tensor(y_val, dtype=torch.long)

        # torchTestX = torch.tensor(X_test_pca)
        # torchTestY = torch.tensor(y_test, dtype=torch.long)

        #ICA
        # transformer = FastICA(n_components=4)
        # X_train_ica = transformer.fit_transform(X_train)
        # X_val_ica = transformer.transform(X_val)
        # X_test_ica = transformer.transform(X_test)

        # torchTrainX = torch.tensor(X_train_ica)
        # torchTrainY = torch.tensor(y_train, dtype=torch.long)

        # torchValX = torch.tensor(X_val_ica)
        # torchValY = torch.tensor(y_val, dtype=torch.long)

        # torchTestX = torch.tensor(X_test_ica)
        # torchTestY = torch.tensor(y_test, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(torchTrainX, torchTrainY)
    test_dataset = torch.utils.data.TensorDataset(torchTestX, torchTestY)
    val_dataset = torch.utils.data.TensorDataset(torchValX, torchValY)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    if neuralNet_paramSearch:

        if fma:
            numClasses = 8
            # numFeats = 518

            # search parameter values
            kern1Sizes = [2, 3, 4, 5]
            kern2Sizes = [2, 3, 4, 5]
            kern3Sizes = [2, 3, 4]

        elif bna:
            numClasses = 2
            numFeats = 4

            # search parameter values
            kern1Sizes = [2]
            kern2Sizes = [2]
            kern3Sizes = [1]

        # data structure to store results
        # netAcc = np.zeros((4,4,3))
        netAcc = np.zeros((len(kern1Sizes), len(kern2Sizes), len(kern3Sizes)))

        numValCombos = len(kern1Sizes) * len(kern2Sizes) * len(kern3Sizes)
        iterCount = 0

        for i in range(len(kern1Sizes)):
            for j in range(len(kern2Sizes)):
                for k in range(len(kern3Sizes)):

                    iterCount += 1

                    kern1 = kern1Sizes[i]
                    kern2 = kern2Sizes[j]
                    kern3 = kern3Sizes[k]

                    #calc the input size to fc layer (518 features per example)
                    fcDim = math.floor(numFeats/kern1)
                    fcDim = math.floor(fcDim/kern3)
                    fcDim = math.floor(fcDim/kern2)
                    fcDim = math.floor(fcDim/kern3)
                    fcDim = fcDim * 20 #num output channels from conv2 layer

                    print('Testing value combo ' + str(iterCount) + ' of ' + str(numValCombos))

                    #need to calculate the shape of the input to fc layer

                    class Net(nn.Module):

                        def __init__(self):
                            super(Net, self).__init__()

                            self.conv1 = nn.Conv1d(1, 10, kernel_size=kern1, stride=kern1)
                            self.conv2 = nn.Conv1d(10, 20, kernel_size=kern2, stride=kern2)
                            self.mp = nn.MaxPool1d(kernel_size=kern3, stride=kern3)
                            self.fc = nn.Linear(fcDim, numClasses)
                            # self.fc = nn.Linear(2520, 8)
                            # self.do = nn.Dropout(p=0.5)

                        def forward(self, x):
                            in_size = x.size(0)

                            x = F.relu(self.mp(self.conv1(x)))
                            x = F.relu(self.mp(self.conv2(x)))
                            # x = self.do(x)

                            # flatten tensor
                            x = x.view(in_size, -1)

                            # fully-connected layer
                            x = self.fc(x)
                            return F.log_softmax(x, dim=1)


                    model = Net()

                    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

                    def train(epoch):
                        model.train()
                        for batch_idx, (data, target) in enumerate(train_loader):
                            # print(data.size())
                            # print(target.size())
                            data, target = Variable(data), Variable(target)
                            data = data.unsqueeze(1) #testing insertion of dimension
                            data = data.float()
                            optimizer.zero_grad()
                            output = model(data)
                            loss = F.nll_loss(output, target)
                            loss.backward()
                            optimizer.step()
                            if batch_idx % 10 == 0:
                                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                    epoch, batch_idx * len(data), len(train_loader.dataset),
                                    100. * batch_idx / len(train_loader), loss.item())) #loss.data[0]


                    def test():
                        model.eval()
                        test_loss = 0
                        correct = 0
                        for data, target in val_loader:
                            # data, target = Variable(data, volatile=True), Variable(target)
                            data, target = Variable(data), Variable(target)
                            data = data.unsqueeze(1) #testing insertion of dimension
                            data = data.float()
                            output = model(data)
                            # sum up batch loss
                            # test_loss += F.nll_loss(output, target, size_average=False).data[0]
                            # test_loss += F.nll_loss(output, target, size_average=False).item()
                            test_loss += F.nll_loss(output, target, reduction='sum').item()
                            # get the index of the max log-probability
                            pred = output.data.max(1, keepdim=True)[1]
                            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                        test_loss /= len(val_loader.dataset)
                        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                            test_loss, correct, len(val_loader.dataset),
                            100. * correct / len(val_loader.dataset)))
                        return 100. * correct / len(val_loader.dataset)

                    for epoch in range(1, 10):
                        train(epoch)
                        testAcc = test()

                    netAcc[i,j,k] = testAcc

        # Save the formatted data:
        with open('neuralNetAccData.pkl', 'wb') as f:
            pickle.dump([netAcc], f)

        with open('neuralNetAccData.pkl', 'rb') as f:
            netAcc = pickle.load(f)
        netAcc = netAcc[0]

        # search parameter values
        # kern1Sizes = [2, 3, 4, 5]
        # kern2Sizes = [2, 3, 4, 5]
        # kern3Sizes = [2, 3, 4]

        #max parameter values
        ind = np.unravel_index(np.argmax(netAcc, axis=None), netAcc.shape)
        print('indices of best params:')
        print(ind)

        if fma:
            f1_f2 = np.squeeze(np.mean(netAcc, axis=2))
            f1_f3 = np.squeeze(np.mean(netAcc, axis=1))
            f2_f3 = np.squeeze(np.mean(netAcc, axis=0))

            #colorbar params
            maxVal = np.amax(netAcc)
            minVal = np.amin(netAcc)

        if bna:
            f1_f2 = np.squeeze(np.mean(netAcc, axis=0))
            f1_f3 = np.squeeze(np.mean(netAcc, axis=0))
            f2_f3 = np.squeeze(np.mean(netAcc, axis=0))

            f1_f2 = np.expand_dims(f1_f2, axis=0)
            f1_f2 = np.expand_dims(f1_f2, axis=1)

            f1_f3 = np.expand_dims(f1_f3, axis=0)
            f1_f3 = np.expand_dims(f1_f3, axis=1)

            f2_f3 = np.expand_dims(f2_f3, axis=0)
            f2_f3 = np.expand_dims(f2_f3, axis=1)

            #colorbar params
            maxVal = 100
            minVal = 0

        # plotting
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(f1_f2, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax1.set_ylabel('conv1 kernel size')
        ax1.set_yticks(np.arange(len(kern1Sizes)))
        ax1.set_yticklabels(kern1Sizes)
        ax1.set_xlabel('conv2 kernel size')
        ax1.set_xticks(np.arange(len(kern2Sizes)))
        ax1.set_xticklabels(kern2Sizes)

        ax2.imshow(f1_f3, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax2.set_ylabel('conv1 kernel size')
        ax2.set_yticks(np.arange(len(kern1Sizes)))
        ax2.set_yticklabels(kern1Sizes)
        ax2.set_xlabel('pool kernel size')
        ax2.set_xticks(np.arange(len(kern3Sizes)))
        ax2.set_xticklabels(kern3Sizes)

        im = ax3.imshow(f2_f3, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax3.set_ylabel('conv2 kernel size')
        ax3.set_yticks(np.arange(len(kern2Sizes)))
        ax3.set_yticklabels(kern2Sizes)
        ax3.set_xlabel('pool kernel size')
        ax3.set_xticks(np.arange(len(kern3Sizes)))
        ax3.set_xticklabels(kern3Sizes)

        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im, cax=cbar_ax)

        f.suptitle('Neural Net Hyperparameter Grid Search')

        plt.show()

    if neuralNet_learningCurve:

        if fma:
            # kern1 = 2
            # kern2 = 5
            # kern3 = 2

            #pca
            # if not part5:
            #     kern1 = 4
            #     kern2 = 4
            #     kern3 = 3
            # elif part5:
            #     kern1 = 4
            #     kern2 = 5
            #     kern3 = 2

            #ica
            # if not part5:
            #     kern1 = 3
            #     kern2 = 3
            #     kern3 = 3
            # elif part5:
            #     kern1 = 2
            #     kern2 = 2
            #     kern3 = 4

            #srp
            # if not part5:
            #     kern1 = 3
            #     kern2 = 2
            #     kern3 = 2
            # elif part5:
            #     kern1 = 4
            #     kern2 = 5
            #     kern3 = 2

            #kb
            if not part5:
                kern1 = 2
                kern2 = 2
                kern3 = 2
            elif part5:
                kern1 = 4
                kern2 = 2
                kern3 = 2

            numClasses = 8
            # numFeats = 518
        elif bna:
            kern1 = 2
            kern2 = 2
            kern3 = 1

            numClasses = 2
            numFeats = 4

        #calc the input size to fc layer (518 features per example)
        fcDim = math.floor(numFeats/kern1)
        fcDim = math.floor(fcDim/kern3)
        fcDim = math.floor(fcDim/kern2)
        fcDim = math.floor(fcDim/kern3)
        fcDim = fcDim * 20 #num output channels from conv2 layer

        class Net(nn.Module):

            def __init__(self):
                super(Net, self).__init__()

                self.conv1 = nn.Conv1d(1, 10, kernel_size=kern1, stride=kern1)
                self.conv2 = nn.Conv1d(10, 20, kernel_size=kern2, stride=kern2)
                self.mp = nn.MaxPool1d(kernel_size=kern3, stride=kern3)
                self.fc = nn.Linear(fcDim, numClasses)
                # self.fc = nn.Linear(2520, 8)
                # self.do = nn.Dropout(p=0.5)

            def forward(self, x):
                in_size = x.size(0)

                x = F.relu(self.mp(self.conv1(x)))
                x = F.relu(self.mp(self.conv2(x)))
                # x = self.do(x)

                # flatten tensor
                x = x.view(in_size, -1)

                # fully-connected layer
                x = self.fc(x)
                return F.log_softmax(x, dim=1)


        model = Net()

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        def train(epoch):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # print(data.size())
                # print(target.size())
                data, target = Variable(data), Variable(target)
                data = data.unsqueeze(1) #testing insertion of dimension
                data = data.float()
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item())) #loss.data[0]


        def test():
            model.eval()
            test_loss = 0
            correct = 0
            for data, target in val_loader:
                # data, target = Variable(data, volatile=True), Variable(target)
                data, target = Variable(data), Variable(target)
                data = data.unsqueeze(1) #testing insertion of dimension
                data = data.float()
                output = model(data)
                # sum up batch loss
                # test_loss += F.nll_loss(output, target, size_average=False).data[0]
                # test_loss += F.nll_loss(output, target, size_average=False).item()
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(val_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(val_loader.dataset),
                100. * correct / len(val_loader.dataset)))
            testAcc = 100. * correct / len(val_loader.dataset)

            test_loss = 0
            correct = 0
            for data, target in train_loader:
                # data, target = Variable(data, volatile=True), Variable(target)
                data, target = Variable(data), Variable(target)
                data = data.unsqueeze(1) #testing insertion of dimension
                data = data.float()
                output = model(data)
                # sum up batch loss
                # test_loss += F.nll_loss(output, target, size_average=False).data[0]
                # test_loss += F.nll_loss(output, target, size_average=False).item()
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            trainAcc = 100. * correct / len(train_loader.dataset)

            return testAcc, trainAcc

        numEpochs = 10
        testTrainAccs = np.zeros((2,numEpochs))
        for epoch in range(1, numEpochs):
            train(epoch)
            testAcc, trainAcc = test()
            testTrainAccs[0, epoch] = testAcc
            testTrainAccs[1, epoch] = trainAcc

        plt.plot(list(range(1,numEpochs+1)), np.squeeze(testTrainAccs[0,:]))
        plt.plot(list(range(1,numEpochs+1)), np.squeeze(testTrainAccs[1,:]))
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend(['Test Accuracy', 'Train Accuracy'])
        plt.title('Neural Net Learning Curve\n' + \
            '(Conv Layer 1 kernel size = ' + str(kern1) + ', Conv Layer 2 kernel size = ' + str(kern2) + ', ' + \
            'Pooling Layer kernel size = ' + str(kern3) + ')')
        plt.show()

    if neuralNet_testAssess:

        if fma:
            # kern1 = 2
            # kern2 = 5
            # kern3 = 2

            #pca
            # if not part5:
            #     kern1 = 4
            #     kern2 = 4
            #     kern3 = 3
            # elif part5:
            #     kern1 = 4
            #     kern2 = 5
            #     kern3 = 2

            #ica
            # if not part5:
            #     kern1 = 3
            #     kern2 = 3
            #     kern3 = 3
            # elif part5:
            #     kern1 = 2
            #     kern2 = 2
            #     kern3 = 4

            #srp
            # if not part5:
            #     kern1 = 3
            #     kern2 = 2
            #     kern3 = 2
            # elif part5:
            #     kern1 = 4
            #     kern2 = 5
            #     kern3 = 2

            #kb
            if not part5:
                kern1 = 2
                kern2 = 2
                kern3 = 2
            elif part5:
                kern1 = 4
                kern2 = 2
                kern3 = 2

            numClasses = 8
            # numFeats = 518
        elif bna:
            kern1 = 2
            kern2 = 2
            kern3 = 1

            numClasses = 2
            numFeats = 4

        #calc the input size to fc layer (518 features per example)
        fcDim = math.floor(numFeats/kern1)
        fcDim = math.floor(fcDim/kern3)
        fcDim = math.floor(fcDim/kern2)
        fcDim = math.floor(fcDim/kern3)
        fcDim = fcDim * 20 #num output channels from conv2 layer

        #need to calculate the shape of the input to fc layer

        class Net(nn.Module):

            def __init__(self):
                super(Net, self).__init__()

                self.conv1 = nn.Conv1d(1, 10, kernel_size=kern1, stride=kern1)
                self.conv2 = nn.Conv1d(10, 20, kernel_size=kern2, stride=kern2)
                self.mp = nn.MaxPool1d(kernel_size=kern3, stride=kern3)
                self.fc = nn.Linear(fcDim, numClasses)
                # self.fc = nn.Linear(2520, 8)
                # self.do = nn.Dropout(p=0.5)

            def forward(self, x):
                in_size = x.size(0)

                x = F.relu(self.mp(self.conv1(x)))
                x = F.relu(self.mp(self.conv2(x)))
                # x = self.do(x)

                # flatten tensor
                x = x.view(in_size, -1)

                # fully-connected layer
                x = self.fc(x)
                return F.log_softmax(x, dim=1)


        model = Net()

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        def train(epoch):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # print(data.size())
                # print(target.size())
                data, target = Variable(data), Variable(target)
                data = data.unsqueeze(1) #testing insertion of dimension
                data = data.float()
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                # if batch_idx % 10 == 0:
                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(data), len(train_loader.dataset),
                #         100. * batch_idx / len(train_loader), loss.item())) #loss.data[0]


        def test():
            model.eval()
            test_loss = 0
            correct = 0
            preds = np.empty((0,1))
            # probs = np.empty((0,8))
            probs = torch.zeros(0, 8)
            for data, target in test_loader:
                # data, target = Variable(data, volatile=True), Variable(target)
                data, target = Variable(data), Variable(target)
                data = data.unsqueeze(1) #testing insertion of dimension
                data = data.float()
                output = model(data)
                probs = torch.cat((probs, output), 0)
                # temp = torch.nn.Softmax(output)
                # print(output)
                # quit()
                # print(output)
                # quit()
                # sum up batch loss
                # test_loss += F.nll_loss(output, target, size_average=False).data[0]
                # test_loss += F.nll_loss(output, target, size_average=False).item()
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                #append preds
                preds = np.append(preds, pred.numpy(), axis=0)
                # preds.
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(test_loader.dataset)
            # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #     test_loss, correct, len(test_loader.dataset),
            #     100. * correct / len(test_loader.dataset)))
            acc = 100. * correct / len(test_loader.dataset)

            return acc, preds, probs

        trainStart = time.time()
        for epoch in range(1, 10):
            train(epoch)
        trainEnd = time.time()

        testStart = time.time()
        testAcc, preds, probs = test()
        testEnd = time.time()


        # clf = tree.DecisionTreeClassifier(max_depth=9, min_samples_leaf=31, \
        #                                   min_samples_split=4)
        # trainStart = time.time()
        # clf = clf.fit(X_train, y_train)
        # trainEnd = time.time()

        # testStart = time.time()
        # score = clf.score(X_test, y_test)
        # testEnd = time.time()

        print('Neural Net accuracy: {:.2%}'.format(testAcc))
        print('Train time: {:.5f}'.format(trainEnd - trainStart))
        print('Test time: {:.5f}'.format(testEnd - testStart))

        if fma:
            uniqueLabels = y_test.unique().tolist()
        elif bna:
            uniqueLabels = ['Genuine', 'Forged']

        if fma:
            cm = confusion_matrix(y_test, le.inverse_transform(preds.astype(int)), labels=uniqueLabels)
        if bna:
            # predictY = convertToStr_bna(predictY)
            cm = confusion_matrix(convertToStr_bna(y_test), convertToStr_bna(preds), labels=uniqueLabels)

        #Confusion Mat
        plt.figure()
        confusionMatrix.plot_confusion_matrix(cm, uniqueLabels, title='Neural Net Confusion Matrix')
        # plt.show()

        #ROC
        sm = torch.nn.Softmax()
        probabilities = sm(probs)
        probabilities = probabilities.detach().numpy()
        if fma:
            skplt.metrics.plot_roc_curve(y_test, probabilities, title='Neural Net ROC Curves')
        if bna:
            skplt.metrics.plot_roc_curve(convertToStr_bna(y_test), probabilities[::-1], title='Neural Net ROC Curves')
        plt.show()
