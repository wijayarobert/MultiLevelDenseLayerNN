# -*- coding: utf-8 -*-
"""
@author: u7177316 - Robert Wijaya
"""
#Import libraries, modules, dataset
import tensorflow as tf
from models.net import model
from dataset.boston import train_dataset, test_dataset, train_labels, test_labels, normed_train_data, normed_test_data
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

#Model declaration
model = model()
optimizer = tf.keras.optimizers.Adam(0.001)
model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])

#TRAINING Part

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

#Plot MAE and MSE during training
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist)

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.title('Mean Absolute Error vs Epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error (MAE)')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,25])
  plt.legend()

  plt.figure()
  plt.title('Mean Square Error vs Epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error (MSE)')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,600])
  plt.legend()
  plt.show()
  
plot_history(history)

#TESTING part

#Get loss value, mae and mse of testing dataset
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MEDV".format(mae))

test_predictions = model.predict(normed_test_data).flatten()
train_predictions = model.predict(normed_train_data).flatten()

#Plot regression graph
plt.scatter(test_labels, test_predictions, color= 'skyblue', edgecolor='navy')
plt.title('Regression graph of testing dataset')
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

#Plot histogram
sns.distplot(test_predictions - test_labels)
plt.title("Histogram of Prediction error")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show() 

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(test_labels, test_predictions)
print('Mean Squared Error: ',mse)
mae = mean_absolute_error(test_labels, test_predictions)
print('Mean Absolute Error: ',mae)
rsq = r2_score(train_labels,train_predictions) #R-Squared on the training data
print('R-square, Training: ',rsq)
rsq = r2_score(test_labels,test_predictions) #R-Squared on the testing data
print('R-square, Testing: ',rsq)

#Show the true value and predicted value
true_predicted = pd.DataFrame(list(zip(test_labels, test_predictions)), 
                    columns=['True Value','Predicted Value'])
print(true_predicted.head(10))