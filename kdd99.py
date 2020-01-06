# -*- coding: utf-8 -*-

"""
Title:  COMP9417, 2019S2  
Name:   z5141730 Bohan Zhao, z5212483 Kunxing Zhang   
Topic:  KDD99 Dataset - Create a Classifier for Intrusion Detection
!! For more details you can look at notbook and report !!
"""

import numpy as np
import pandas as pd
from tensorflow.keras.utils import get_file
from sklearn.model_selection import train_test_split
from sklearn import metrics

try:
    path = get_file('kddcup.data_10_percent.gz', origin = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')
except:
    print('Error downloading')
    raise

# Download from: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
df = pd.read_csv(path, header = None)

print("This file datasets have {} rows.\n".format(len(df)))

df.dropna(inplace = True,axis = 1)

df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]

df.head()

numberic_feature = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 
                   'urgent', 'hot', 'num_failed_logins', 'num_compromised',
                   'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                   'num_shells', 'num_access_files', 'num_outbound_cmds',
                   'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                   'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                   'dst_host_srv_count', 'dst_host_same_srv_rate',
                   'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                   'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                   'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                   'dst_host_srv_rerror_rate']

categirical_features = ['protocol_type', 'service', 'flag', 'land', 'logged_in',
                       'is_host_login', 'is_guest_login']

for f in numberic_feature:
  df[f] = (df[f] - df[f].mean()) / df[f].std()

for f in categirical_features:
  dummies = pd.get_dummies(df[f])
  
  for x in dummies.columns:
    df[f"{f}-{x}"] = dummies[x]
  df.drop(f, axis = 1, inplace = True)

df.dropna(inplace = True, axis = 1)

df.head()

x_columns = df.columns.drop('outcome')
x = df[x_columns].values
dummies = pd.get_dummies(df['outcome']) # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y = dummies.values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.25, random_state = 42)

# DNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

print("DNN:")
model = Sequential()
model.add(Dense(10, input_dim = x.shape[1], kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(50, input_dim = x.shape[1], kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(10, input_dim = x.shape[1], kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'normal'))
model.add(Dense(y.shape[1],activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
monitor = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = 5, verbose = 1, mode = 'auto')
model.fit(x_train, y_train, validation_data = (x_test, y_test), callbacks = [monitor], verbose = 2, epochs = 100)

pred = model.predict(x_test)
pred = np.argmax(pred,axis = 1)
y_eval = np.argmax(y_test,axis = 1)
score = metrics.accuracy_score(y_eval, pred)
print("Accuracy: {}\n".format(score))

# Random Forest
from sklearn.ensemble import RandomForestClassifier

print("Random Forest:")
rfc = RandomForestClassifier(n_estimators = 100, random_state = 90)
rfc.fit(x_train,y_train)
score = rfc.score(x_test,y_test)
print("Accuracy: %s\n"%score)

# k-NN
from sklearn.neighbors import KNeighborsClassifier

print("k-NN:")
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
score = knn.score(x_test,y_test)
print("Accuracy: %s\n"%score)