import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#import tf.keras
import math
import time
import numpy as np
batch = 10
epoch = 50

p=[0.1]
q=[0.8]

learning_rate=0.001


#Read data
data = pd.read_csv("adult.csv").dropna()
data=data.drop(['fnlwgt'],axis=1)

data['relationship']=np.where(data['relationship'] ==' Not-in-family', 'Yes', data['relationship'])
data['relationship']=np.where(data['relationship'] ==' Husband', 'Yes', data['relationship'])
data['relationship']=np.where(data['relationship'] ==' Wife', 'Yes', data['relationship'])
data['relationship']=np.where(data['relationship'] ==' Own-child', 'Yes', data['relationship'])
data['relationship']=np.where(data['relationship'] ==' Other-relative', 'Yes', data['relationship'])
data['relationship']=np.where(data['relationship'] ==' Unmarried', 'No', data['relationship'])

train = data[:39040]
test = data[39040:48832]

#Preprocess Data - convert string to numerical data
cat_feats = [column for column in data.columns if data[column].dtypes=='object']
non_cat_feats = [column for column in data.columns if data[column].dtypes!='object']

train[cat_feats] = train[cat_feats].apply(LabelEncoder().fit_transform)
test[cat_feats] = test[cat_feats].apply(LabelEncoder().fit_transform)

X_train = train.iloc[:,0:-1]
y_train = train[['income']]

X_test = test.iloc[:,0:-1]
y_test = test[['income']]

scaler = StandardScaler()

scaled_train = StandardScaler().fit_transform(X_train[non_cat_feats].values)
scaled_test = StandardScaler().fit_transform(X_test[non_cat_feats].values)

X_train[non_cat_feats] = scaled_train
X_test[non_cat_feats] = scaled_test

def calculateEps(p,q):
    numerator=(p+(1-p)*q)
    denominator=(1-p)*q
    eps=math.log(numerator/denominator)
    return eps

def rr(column,p,q):
    for i in range(column.count()):
        flip1 = random.random() < p
        if flip1:
            continue
        else:
            flip2 = random.random() < q
            if flip2:
                column[i] = 1
            else:
                column[i] = 0

for i,j in zip(p,q):
    accuracy=0
    #rr(y_train['income'],i,j)

    rr(X_train['sex'],i,j)
    rr(X_train['relationship'],i,j)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dropout(rate=0.2, input_shape=X_train.shape[1:]))
    for _ in range(2):
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print("We are using Tensorflow version", tf.__version__)
    print("tf.keras API version: {}".format(tf.keras.__version__))
    #es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    print("for p=% and q=%",(i,j))
    print(calculateEps(i,j))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.Accuracy()])
    for k in range(1,11):
        start = time.time()
        print("=======================Iteration=====================",k)
        history = model.fit(X_train, y_train,
             batch_size=batch,
             validation_data=(X_test, y_test),
             epochs=epoch)
             #callbacks=[es_cb])
        print('\nIt took {} seconds'.format(time.time()-start))
        acc=history.history['accuracy'][49]
        if(acc > accuracy):
            print("Saving accuracy as previous accuracy was {} and present is {}" .format(accuracy,acc))
            modelName = "RRmodelDP_.13.h5"
            model.save(modelName)
            accuracy=acc







#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#print(X_train.shape)

'''
X=df[['age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']]
y=df[['income']]'''

