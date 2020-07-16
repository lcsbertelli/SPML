import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# import tf.keras
import math
import time
import numpy as np

batch = 10
epoch = 50


learning_rate = 0.001

# Read data
data = pd.read_csv("adult.csv").dropna()
data = data.drop(['fnlwgt'], axis=1)

train = data[:39040]
test = data[39040:78080]

# Preprocess Data - convert string to numerical data
cat_feats = [column for column in data.columns if data[column].dtypes == 'object']
non_cat_feats = [column for column in data.columns if data[column].dtypes != 'object']

train[cat_feats] = train[cat_feats].apply(LabelEncoder().fit_transform)
test[cat_feats] = test[cat_feats].apply(LabelEncoder().fit_transform)

X_train = train.iloc[:, 0:-1]
y_train = train[['income']]

X_test = test.iloc[:, 0:-1]
y_test = test[['income']]

scaler = StandardScaler()

scaled_train = StandardScaler().fit_transform(X_train[non_cat_feats].values)
scaled_test = StandardScaler().fit_transform(X_test[non_cat_feats].values)

X_train[non_cat_feats] = scaled_train
X_test[non_cat_feats] = scaled_test

def train():
    accuracy=0
    # rr(y_train['income'],i,j)

    # rr(X_train['sex'],i,j)
    # rr(X_train['relationship'],i,j)

    print("We are using Tensorflow version", tf.__version__)
    print("tf.keras API version: {}".format(tf.keras.__version__))
    for k in range(1,11):
        print("=======================Iteration=====================",k)
        start = time.time()
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dropout(rate=0.2, input_shape=X_train.shape[1:]))
        for _ in range(2):
            model.add(tf.keras.layers.Dense(units=64, activation='relu'))
            model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        #es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.Accuracy()])
        history = model.fit(X_train, y_train,
             batch_size=batch,
             validation_data=(X_test, y_test),
             epochs=epoch)
             #callbacks=[es_cb])
        print('\nIt took {} seconds'.format(time.time()-start))
        acc=history.history['accuracy'][49]
        if(acc > accuracy):
            print("Saving accuracy as previous accuracy was {} and present is {}" .format(accuracy,acc))
            modelName = "RRmodel" + "native" + ".h5"
            model.save(modelName)
            accuracy=acc


train()

