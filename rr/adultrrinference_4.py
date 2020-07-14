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

    modelName = "RRmodelDP_4.h5"
    model = tf.keras.models.load_model("rrnative/"+modelName)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.Accuracy()])

    for k in range(1, 11):
        start_time = time.time()
        # Compile model with Keras
        print('\n# Evaluate on test data')
        print("==================================Iteration %============================",k)
        results = model.evaluate(X_test, y_test, batch_size=batch)
        #print('test loss, test acc:', results)
        print("Accuracy:",results[1]*100)
        print("Latency --- %s seconds ---" % (time.time() - start_time))

train()

