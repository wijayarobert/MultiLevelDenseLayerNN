# -*- coding: utf-8 -*-
"""
@author: u7177316 - Robert Wijaya
"""
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import BatchNormalization
from keras.layers import concatenate

def model():
  inputs = keras.Input(shape=13)

  inputs_normalize = BatchNormalization(axis=1)(inputs)

  NN1_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
  NN2_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
  NN3_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
  NN4_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
  NN5_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
  NN6_Dense = layers.Dense(128, activation='relu')(inputs_normalize)

  merge1 = concatenate([NN1_Dense, NN2_Dense])
  NN_merge1 = layers.Dense(128, activation='relu')(merge1)
  merge2 = concatenate([NN3_Dense, NN4_Dense])
  NN_merge2 = layers.Dense(128, activation='relu')(merge2)
  merge3 = concatenate([NN5_Dense, NN6_Dense])
  NN_merge3 = layers.Dense(128, activation='relu')(merge3)

  merge_all = concatenate([NN_merge1, NN_merge2, NN_merge3])

  output = layers.Dense(1)(merge_all)

  return keras.Model(inputs=inputs, outputs=output)