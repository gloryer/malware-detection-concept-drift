import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


from spektral.data import Dataset, DisjointLoader, Graph
from spektral.datasets import TUDataset
from spektral.layers import GINConv, GlobalAvgPool
from spektral.transforms.normalize_adj import NormalizeAdj

import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix


import os
import matplotlib.pyplot as plt
import datetime
import math
import pandas as pd
import pickle as pkl
from PIL import Image
import random as python_random

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Layer, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.layers import MaxPool2D,  BatchNormalization, Activation, Conv2DTranspose,Reshape, UpSampling2D
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras import backend as K
from keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from sklearn.metrics import f1_score



from IPython import display
from matplotlib import pyplot
from matplotlib.pyplot import figure
from numpy import expand_dims
import random

from tensorflow.keras.utils import get_source_inputs

from utils import *


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()






class list_to_spektral_dataset(Dataset):
    def __init__(self, data, **kwargs):
        self.data = data

        super().__init__(**kwargs)
        
    def read(self):
        #print(self.data[0])
        return [Graph(x=graph.x, a=graph.a, y=graph.y) for graph in self.data]


def filter_label(dataset, label):
    #divide dataset based on the label[]
     
    output = []
    res = []
    
    for graph in dataset: 
        if graph.y in label:
            output.append(graph)
        else: 
            res.append(graph)
            
    output_dataset = list_to_spektral_dataset(output)
    res_dataset = list_to_spektral_dataset(res)
    return  output_dataset, res_dataset

def train_test_split(dataset, train_percentage):
    # Train/test split
    idxs = np.random.permutation(len(dataset)) 
    split = int(train_percentage * len(dataset))
    idx_tr, idx_te = np.split(idxs, [split])
    #print(bool(set(idx_tr) & set(idx_te)))
    dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]
    
    return dataset_tr, dataset_te


def subsample(dataset, size):
    # Train/test split
    idxs = np.random.permutation(len(dataset)) 
    split = int(size)
    idx_sample, idx_re = np.split(idxs, [split])
    dataset_sample, dataset_re= dataset[idx_sample], dataset[idx_re]
    
    return dataset_sample, dataset_re

def merge_dataset(dataset1, dataset2): 
    # implemented fucntions in Dataset class of spektral graph 
    merged = dataset1. __add__(dataset2)
    return merged 


def binary_label(dataset, flag_attack):
    
    for g in dataset: 
        #g.y = np.pad(g.y, (0, 1), 'constant')
        if flag_attack: 
            y = np.zeros((2,))
            y[1] = 1
            g.y = y
        else:
            y = np.zeros((2,))
            y[0] = 1
            g.y = y
        
    return dataset

def convert_label_integer(dataset):
    for g in dataset: 
        #g.y = np.pad(g.y, (0, 1), 'constant')
        g.y = np.where(g.y==1.)[0][0] 
        
    return dataset

def update_labels(dataset, path):
    
    
    labels = np.load(path)
    
    if len(labels)!=len(dataset):
        print("Label size {} does not match data size{}".format(len(labels),len(dataset)))
           
    i = 0 
    for g in dataset:
        g.y=labels[i]
        i+=1
        
    return dataset
    


def obtain_labels(dataset):
    
    y = []
    
    for g in dataset:
        y.append(g.y)
    
    return np.array(y)


def obtain_feature_matrix(dataset):

    graph_x = []
    

    
    for g in dataset:
        graph_x.append(np.sum(g.x, axis=0))
    return np.array(graph_x)



def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
