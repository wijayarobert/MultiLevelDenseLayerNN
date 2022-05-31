# -*- coding: utf-8 -*-
"""
@author: u7177316 - Robert Wijaya
"""
from sklearn.datasets import load_boston
import pandas as pd
import tensorflow as tf

#Load dataset
boston_dataset = load_boston()

#Convert into pandas dataframe
dataset = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
dataset['MEDV'] = pd.Series(data=boston_dataset.target, index=dataset.index)

#Initialize dataset and store it into variable

def train_dataset(dataset):
    '''
    This function will split the original dataset for training (80%)
    Parameters
    ----------
    dataset : A dataframe.
              The original Boston dataset.

    Returns
    -------
    train_dataset : A dataframe.
                    A dataset for training.
    '''
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    return train_dataset

def test_dataset(dataset):
    test_dataset = dataset.drop(train_dataset.index)
    return test_dataset

train_dataset = train_dataset(dataset)
test_dataset = test_dataset(dataset)

def train_labels(train_dataset):
    '''
    This function will get the entire row of target column (price) 
    of Boston dataset.

    Parameters
    ----------
    train_dataset : A dataframe.
                    The training dataset of Boston dataset.

    Returns
    -------
    train_labels : A dataframe (1 column)
                   The entire row of target column (price) of Boston dataset.

    '''
    train_labels = train_dataset.pop('MEDV')
    return train_labels

def test_labels(test_dataset):
    test_labels = test_dataset.pop('MEDV')
    return test_labels

train_labels = train_labels(train_dataset)
test_labels = test_labels(test_dataset)

def normalized_train_data(train_dataset):
    '''
    This function will normalize the training dataset with z-score method
    using Tensorflow inbuilt normalize function.

    Parameters
    ----------
    train_dataset : A dataframe.
                    A training dataset of Boston Dataset.

    Returns
    -------
    normed_train_data : A dataframe.
                        Normalized dataset.

    '''
    normed_train_data = tf.keras.utils.normalize(train_dataset)
    return normed_train_data
    
def normalized_test_data(test_dataset):
    normed_test_data = tf.keras.utils.normalize(test_dataset)
    return normed_test_data   

#Normalize training and testing dataset    
normed_train_data = tf.keras.utils.normalize(train_dataset)
normed_test_data = tf.keras.utils.normalize(test_dataset)




