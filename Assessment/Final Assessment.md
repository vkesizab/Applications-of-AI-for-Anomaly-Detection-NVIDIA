<a href="https://www.nvidia.com/en-us/deep-learning-ai/education/"> <img src="images/DLI Header.png" alt="Header" style="width: 400px;"/> </a>

# Workshop Assessment 

Welcome to the assessment section of this course. In the previous labs you successfully applied machine learning and deep learning techniques for the task of anomaly detection on network packet data. Equipped with this background, you can apply these techniques to any type of data (images or audio) across different use cases. In this assessment, you will apply supervised and unsupervised techniques for intrusion detection on the NSL KDD dataset.

If you are successfully able to complete this assessment, you will be able to generate a certificate of competency for the course. Good luck!

## Objectives

This assessment seeks to test the following concepts:

1.   Building and training an Xgboost model.
2.   Building and training an autoencoder neural network.
3.   Detecting anomalies using different thresholding methods.

The total duration of the assessment is 2 hrs, however, if you are unable to complete the assessment today, you are more than welcome to return to it at a later time to try and complete it then.

## Section 1: Preparation - Done for You

### The Dataset

We will be using the NSL-KDD dataset published by the University of New Brunswick in this assessment. While the dataset is similar to the KDD dataset used throughout the workshop in terms of the features used, it varies in the following respects:

1.   Removal of redundant and duplicate records in the dataset to prevent classifiers from overfitting a particular class.
2.   The number of selected records from each difficulty level group is inversely proportional to the percentage of records in the original KDD data set making the task of unsupervised classification slightly more challenging.

### Imports


```python
import numpy as np
import pandas as pd
import os
import random as python_random

import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import optimizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import plot_model

from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report,roc_curve

# We will use our own accuracy score functions for the sake of grading this assessment
from assessment import xgb_accuracy_score, autoencoder_accuracy_score

from tensorflow.keras.models import load_model, model_from_json

np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED']=str(42)
```

### Load the Data


```python
col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
             "num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
             "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
             "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
             "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
             "dst_host_srv_rerror_rate","label"]

df = pd.read_csv("data/KDDTrain+_20Percent.txt", header=None, names=col_names, index_col=False)

text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>protocol_type</th>
      <th>service</th>
      <th>flag</th>
      <th>src_bytes</th>
      <th>dst_bytes</th>
      <th>land</th>
      <th>wrong_fragment</th>
      <th>urgent</th>
      <th>hot</th>
      <th>...</th>
      <th>dst_host_srv_count</th>
      <th>dst_host_same_srv_rate</th>
      <th>dst_host_diff_srv_rate</th>
      <th>dst_host_same_src_port_rate</th>
      <th>dst_host_srv_diff_host_rate</th>
      <th>dst_host_serror_rate</th>
      <th>dst_host_srv_serror_rate</th>
      <th>dst_host_rerror_rate</th>
      <th>dst_host_srv_rerror_rate</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>tcp</td>
      <td>ftp_data</td>
      <td>SF</td>
      <td>491</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>25</td>
      <td>0.17</td>
      <td>0.03</td>
      <td>0.17</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>udp</td>
      <td>other</td>
      <td>SF</td>
      <td>146</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.60</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>tcp</td>
      <td>private</td>
      <td>S0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>26</td>
      <td>0.10</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>neptune</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>tcp</td>
      <td>http</td>
      <td>SF</td>
      <td>232</td>
      <td>8153</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>255</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>tcp</td>
      <td>http</td>
      <td>SF</td>
      <td>199</td>
      <td>420</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>255</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>




```python
# Describe the different classes of Labels

pd.DataFrame(df['label'].value_counts())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>normal</th>
      <td>13449</td>
    </tr>
    <tr>
      <th>neptune</th>
      <td>8282</td>
    </tr>
    <tr>
      <th>ipsweep</th>
      <td>710</td>
    </tr>
    <tr>
      <th>satan</th>
      <td>691</td>
    </tr>
    <tr>
      <th>portsweep</th>
      <td>587</td>
    </tr>
    <tr>
      <th>smurf</th>
      <td>529</td>
    </tr>
    <tr>
      <th>nmap</th>
      <td>301</td>
    </tr>
    <tr>
      <th>back</th>
      <td>196</td>
    </tr>
    <tr>
      <th>teardrop</th>
      <td>188</td>
    </tr>
    <tr>
      <th>warezclient</th>
      <td>181</td>
    </tr>
    <tr>
      <th>pod</th>
      <td>38</td>
    </tr>
    <tr>
      <th>guess_passwd</th>
      <td>10</td>
    </tr>
    <tr>
      <th>warezmaster</th>
      <td>7</td>
    </tr>
    <tr>
      <th>buffer_overflow</th>
      <td>6</td>
    </tr>
    <tr>
      <th>imap</th>
      <td>5</td>
    </tr>
    <tr>
      <th>rootkit</th>
      <td>4</td>
    </tr>
    <tr>
      <th>phf</th>
      <td>2</td>
    </tr>
    <tr>
      <th>multihop</th>
      <td>2</td>
    </tr>
    <tr>
      <th>loadmodule</th>
      <td>1</td>
    </tr>
    <tr>
      <th>spy</th>
      <td>1</td>
    </tr>
    <tr>
      <th>land</th>
      <td>1</td>
    </tr>
    <tr>
      <th>ftp_write</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Data Preprocessing


```python
# Create one-hot encoded categorical columns in the dataset

cat_vars = ['protocol_type', 'service', 'flag', 'land', 'logged_in','is_host_login', 'is_guest_login']

# Find unique labels for each category
cat_data = pd.get_dummies(df[cat_vars])

# Check that the categorical variables were created correctly
cat_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>land</th>
      <th>logged_in</th>
      <th>is_host_login</th>
      <th>is_guest_login</th>
      <th>protocol_type_icmp</th>
      <th>protocol_type_tcp</th>
      <th>protocol_type_udp</th>
      <th>service_IRC</th>
      <th>service_X11</th>
      <th>service_Z39_50</th>
      <th>...</th>
      <th>flag_REJ</th>
      <th>flag_RSTO</th>
      <th>flag_RSTOS0</th>
      <th>flag_RSTR</th>
      <th>flag_S0</th>
      <th>flag_S1</th>
      <th>flag_S2</th>
      <th>flag_S3</th>
      <th>flag_SF</th>
      <th>flag_SH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 84 columns</p>
</div>




```python
# Separate the numerical columns

numeric_vars = list(set(df.columns.values.tolist()) - set(cat_vars))
numeric_vars.remove('label')
numeric_data = df[numeric_vars].copy()

# Check that the numeric data has been captured accurately

numeric_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>urgent</th>
      <th>num_compromised</th>
      <th>dst_bytes</th>
      <th>num_file_creations</th>
      <th>duration</th>
      <th>same_srv_rate</th>
      <th>dst_host_srv_count</th>
      <th>dst_host_srv_diff_host_rate</th>
      <th>src_bytes</th>
      <th>rerror_rate</th>
      <th>...</th>
      <th>dst_host_count</th>
      <th>srv_count</th>
      <th>num_shells</th>
      <th>count</th>
      <th>num_failed_logins</th>
      <th>dst_host_srv_rerror_rate</th>
      <th>srv_diff_host_rate</th>
      <th>num_outbound_cmds</th>
      <th>root_shell</th>
      <th>srv_rerror_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.00</td>
      <td>25</td>
      <td>0.00</td>
      <td>491</td>
      <td>0.0</td>
      <td>...</td>
      <td>150</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.08</td>
      <td>1</td>
      <td>0.00</td>
      <td>146</td>
      <td>0.0</td>
      <td>...</td>
      <td>255</td>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.05</td>
      <td>26</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>255</td>
      <td>6</td>
      <td>0</td>
      <td>123</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>8153</td>
      <td>0</td>
      <td>0</td>
      <td>1.00</td>
      <td>255</td>
      <td>0.04</td>
      <td>232</td>
      <td>0.0</td>
      <td>...</td>
      <td>30</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>420</td>
      <td>0</td>
      <td>0</td>
      <td>1.00</td>
      <td>255</td>
      <td>0.00</td>
      <td>199</td>
      <td>0.0</td>
      <td>...</td>
      <td>255</td>
      <td>32</td>
      <td>0</td>
      <td>30</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python
numeric_cat_data = pd.concat([numeric_data, cat_data], axis=1)
```

## Assessment Task 1: Data Selection

The first part of this assessment checks whether you understand the data you are working with. If successful, you should be able to load and split the data in order to begin learning from it.

In the code block below, replace each #### FIX ME #### with solutions which:
1.   Determine the number of classes in the dataset.
2.   Set the variable test_size to the fraction of the dataset you would like to use for testing.


```python
# Capture the labels
labels = df['label'].copy()

# Convert labels to integers
le = LabelEncoder()
integer_labels = le.fit_transform(labels)
num_labels = len(np.unique(labels)) ## fixed

# Split data into test and train
x_train, x_test, y_train, y_test = train_test_split(numeric_cat_data,
                                                    integer_labels,
                                                    test_size= 0.25, #### FIXED  ####
                                                    random_state= 42)
```


```python
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
```

    (18894, 118)
    (18894,)
    (6298, 118)
    (6298,)



```python
# Make sure to only fit the the scaler on the training data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Convert the data to FP32
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
```

## Assessment Task 2 : XGBoost - Set the XGBoost Parameters

Treat the question as a **multi-class** supervised learning problem and train a **GPU-accelerated XGBoost model** on the given dataset. Refer to the [documentation](https://xgboost.readthedocs.io/en/latest/parameter.html) or your previous tasks to fix the parameter list. You may reference the notebooks from previous sections by opening the file explorer on the left-hand side of the JupyterLab screen.

This task checks that you know how these parameters impact training.


```python
 params = {
    'num_rounds':        10,   
    'max_depth':         8,
    'max_leaves':        2**8,
    'alpha':             0.9,
    'eta':               0.1,
    'gamma':             0.1,
    'learning_rate':     0.1,
    'subsample':         1,
    'reg_lambda':        1,
    'scale_pos_weight':  2,
    'tree_method':       "gpu_hist",
    'n_gpus':            1,
    'objective':         "multi:softprob",
    'num_class':         22,
    'verbose':           True
}
```

## Assessment Task 3: Model Training

In this next task, you will prove that you can build and fit an accelerated XGBoost Model.
1.   Initiate training by referring to the [XGBoost API](https://xgboost.readthedocs.io/en/latest/python/python_api.html) documentation.
2.   Fit the model on test data to obtain the predictions.


```python
y_train.shape
```




    (18894,)




```python
y_test.shape
```




    (6298,)




```python
x_test.shape
```




    (6298, 118)




```python
x_train.shape
```




    (18894, 118)




```python
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
evals = [(dtest, 'test',), (dtrain, 'train')]

num_rounds = params['num_rounds']

model = xgb.train(params,dtrain,num_rounds,evals=evals) #### 
```

    [01:10:52] WARNING: ../src/learner.cc:576: 
    Parameters: { "n_gpus", "num_rounds", "scale_pos_weight", "verbose" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
    [01:10:53] WARNING: ../src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    [0]	test-mlogloss:2.06695	train-mlogloss:2.06747
    [1]	test-mlogloss:1.70080	train-mlogloss:1.70039
    [2]	test-mlogloss:1.45067	train-mlogloss:1.44993
    [3]	test-mlogloss:1.26043	train-mlogloss:1.25919
    [4]	test-mlogloss:1.10766	train-mlogloss:1.10599
    [5]	test-mlogloss:0.98088	train-mlogloss:0.97874
    [6]	test-mlogloss:0.87355	train-mlogloss:0.87109
    [7]	test-mlogloss:0.78130	train-mlogloss:0.77860
    [8]	test-mlogloss:0.70101	train-mlogloss:0.69816
    [9]	test-mlogloss:0.63058	train-mlogloss:0.62760



```python
preds = model.predict(dtest)
print(preds)

true_labels = y_test
np.save('y_test.npy', y_test)   #
np.save('y_train.npy', y_train) # 
true_labels
```

    [[0.03265708 0.03271218 0.03266538 ... 0.03265694 0.0326893  0.03267832]
     [0.01884467 0.01887646 0.01884945 ... 0.01884458 0.01886326 0.01885692]
     [0.02149509 0.02153136 0.02150055 ... 0.021495   0.0215163  0.02150907]
     ...
     [0.0214987  0.02153497 0.02150416 ... 0.02149861 0.02151991 0.02151268]
     [0.0214987  0.02153497 0.02150416 ... 0.02149861 0.02151991 0.02151268]
     [0.0214987  0.02153497 0.02150416 ... 0.02149861 0.02151991 0.02151268]]





    array([17, 11,  9, ...,  9,  9,  9])




```python
# If predictions > 0.5, pred_labels = 1 else pred_labels = 0

pred_labels = np.argmax(preds, axis=1) #
pred_labels
```




    array([17, 11,  9, ...,  9,  9,  9])



Get the accuracy score for your model's predictions. In order to pass this part of the assessment, you need to attain an accuracy greater than 90%.


```python
# NOTE: We are using our own accuracy score function in order to help grade the assessment,
# though it will behave here exactly like its scikit-learn couterpart `accuracy_score`.
xgb_acc = xgb_accuracy_score(true_labels, pred_labels)
# print ('XGBoost Accuracy Score :', accuracy_score(xgb_acc))

print ('XGBoost Accuracy Score :', xgb_accuracy_score(true_labels, pred_labels))
```

    XGBoost Accuracy Score : 0.9947602413464592


## Assessment Task 4: Implement a Confusion Matrix

Show that you can determine the performance of your model by implementing a confusion matrix.


```python
cm = confusion_matrix(true_labels, pred_labels)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Greens):
    plt.figure(figsize=(10,10),)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(cm)
```


![png](output_38_0.png)



```python
pickle.dump(model, open("xgboost.pickle.dat", "wb"))
```

# Autoencoder Model

As the second major part of this assessment, you get to train your own autoencoder neural network to understand inherant clusters in your data. Build an autoencoder treating this as a brinary classification problem. Feel free to open the file viewer on the left of the JupyterLab environment to view the notebooks from previous sections if you need a reference to guide your work.

![alt text](https://drive.google.com/uc?id=1gexBTwBnK_LtTmxrZp_opHD1xaAd2oum)

## Assessment Task 5: Set the Hyperparameters 


```python
input_dim = x_train.shape[1]

# Model hyperparameters
batch_size = 512

# Latent dimension: higher values add network capacity 
# while lower values increase efficiency of the encoding
latent_dim = 4

# Number of epochs: should be high enough for the network to learn from the data, 
# but not so high as to overfit the training data or diverge
max_epochs = 10 # Or 12

learning_rate = .00001
```

## Assessment Task 6: Build the Encoder Segment

1.   Fix the dimensions of the input (number of features in the dataset) in the input layer.
2.   Define the hidden layers of the encoder. We recommended using at least 3-4 layers.
3.   Consider adding dropout layers to the encoder to help avoid overfitting.
4.   Experiment with different activation functions (relu, tanh, sigmoid etc.).

Feel free to open the file viewer on the left of the JupyterLab environment to view the notebooks from previous sections if you need a reference to guide your work.


```python
print(input_dim)
```

    118



```python
# The encoder will consist of a number of dense layers that decrease in size 
# as we taper down towards the bottleneck of the network: the latent space.

input_data = Input(shape=(input_dim,), name='encoder_input')
    
    
# Hidden layers
# encoder = Dense(units= #### FIX ME ####, activation= 'relu', name= #### FIX ME ####)(input_data)
                  
encoder = Dense(units= 96,activation='tanh', name='encoder_1')(input_data)
encoder = Dropout(.1)(encoder)
encoder = Dense(units= 64,activation='tanh', name='encoder_2')(encoder)
encoder = Dropout(.1)(encoder)
encoder = Dense(units= 48,activation='tanh', name='encoder_3')(encoder)
encoder = Dropout(.1)(encoder)
encoder = Dense(units= 16,activation='tanh', name='encoder_4')(encoder)
encoder = Dropout(.1)(encoder)                
                
# Make your Encoder Deeper

# Bottleneck layer
# latent_encoding = Dense(#### FIX ME ####, activation='linear', name='latent_encoding')(encoder)

latent_encoding = Dense(latent_dim, activation='linear', name='latent_encoding')(encoder)
```


```python
# We instantiate the encoder model, look at a summary of it's layers, and visualize it.

encoder_model = Model(input_data, latent_encoding)

encoder_model.summary()
```

    Model: "model_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    encoder_input (InputLayer)   [(None, 118)]             0         
    _________________________________________________________________
    encoder_1 (Dense)            (None, 96)                11424     
    _________________________________________________________________
    dropout_20 (Dropout)         (None, 96)                0         
    _________________________________________________________________
    encoder_2 (Dense)            (None, 64)                6208      
    _________________________________________________________________
    dropout_21 (Dropout)         (None, 64)                0         
    _________________________________________________________________
    encoder_3 (Dense)            (None, 48)                3120      
    _________________________________________________________________
    dropout_22 (Dropout)         (None, 48)                0         
    _________________________________________________________________
    encoder_4 (Dense)            (None, 16)                784       
    _________________________________________________________________
    dropout_23 (Dropout)         (None, 16)                0         
    _________________________________________________________________
    latent_encoding (Dense)      (None, 4)                 68        
    =================================================================
    Total params: 21,604
    Trainable params: 21,604
    Non-trainable params: 0
    _________________________________________________________________



```python
plot_model(
    encoder_model, 
    to_file='encoder_model.png', 
    show_shapes=True, 
    show_layer_names=True, 
    rankdir='TB' # TB for top to bottom, LR for left to right
)

Image(filename='encoder_model.png')
```

    Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work.



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-69-637453f64104> in <module>
          7 )
          8 
    ----> 9 Image(filename='encoder_model.png')
    

    NameError: name 'Image' is not defined


## Assessment Task 7: Build Decoder Segment

1.    Fix the dimensions of the input to the decoder. 
2.    Grow the network from the latent layer to the output layer of size equal to the input layer.
3.    Experiment with different activation functions (tanh, relu, sigmoid etc.).


```python
# The decoder network is a mirror image of the encoder network.
#decoder = Dense(units = #### FIX ME ####, activation= #### FIX ME ####, name='decoder_1')(latent_encoding)

decoder = Dense(16, activation='tanh', name='decoder_1')(latent_encoding)#Replace with the layer name that feeds this one
decoder = Dropout(.1)(decoder)
decoder = Dense(48, activation='tanh', name='decoder_2')(decoder)#Replace with the layer name that feeds this one
decoder = Dropout(.1)(decoder)
decoder = Dense(64, activation='tanh', name='decoder_3')(decoder)#Replace with the layer name that feeds this one
decoder = Dropout(.1)(decoder)
decoder = Dense(96, activation='tanh', name='decoder_4')(decoder)#Replace with the layer name that feeds this one
decoder = Dropout(.1)(decoder)
                
# The output is the same dimension as the input data we are reconstructing.
                           
reconstructed_data = Dense(units = input_dim, activation='linear', name='reconstructed_data')(decoder)#
```


```python
# We instantiate the encoder model, look at a summary of its layers, and visualize it.
autoencoder_model = Model(input_data, reconstructed_data)

autoencoder_model.summary()
```

    Model: "model_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    encoder_input (InputLayer)   [(None, 118)]             0         
    _________________________________________________________________
    encoder_1 (Dense)            (None, 96)                11424     
    _________________________________________________________________
    dropout_12 (Dropout)         (None, 96)                0         
    _________________________________________________________________
    encoder_2 (Dense)            (None, 64)                6208      
    _________________________________________________________________
    dropout_13 (Dropout)         (None, 64)                0         
    _________________________________________________________________
    encoder_3 (Dense)            (None, 48)                3120      
    _________________________________________________________________
    dropout_14 (Dropout)         (None, 48)                0         
    _________________________________________________________________
    encoder_4 (Dense)            (None, 16)                784       
    _________________________________________________________________
    dropout_15 (Dropout)         (None, 16)                0         
    _________________________________________________________________
    latent_encoding (Dense)      (None, 4)                 68        
    _________________________________________________________________
    decoder_1 (Dense)            (None, 16)                80        
    _________________________________________________________________
    dropout_16 (Dropout)         (None, 16)                0         
    _________________________________________________________________
    decoder_2 (Dense)            (None, 48)                816       
    _________________________________________________________________
    dropout_17 (Dropout)         (None, 48)                0         
    _________________________________________________________________
    decoder_3 (Dense)            (None, 64)                3136      
    _________________________________________________________________
    dropout_18 (Dropout)         (None, 64)                0         
    _________________________________________________________________
    decoder_4 (Dense)            (None, 96)                6240      
    _________________________________________________________________
    dropout_19 (Dropout)         (None, 96)                0         
    _________________________________________________________________
    reconstructed_data (Dense)   (None, 118)               11446     
    =================================================================
    Total params: 43,322
    Trainable params: 43,322
    Non-trainable params: 0
    _________________________________________________________________


## Assessment Task 8: Initiate Training of the Model

1.   Fix the learning rate *Hint: Think in the order of 10e-4*.
2.   Choose an appropriate error metric for the loss function (mse, rmse, mae etc.).
3.   Think about whether you want to shuffle your dataset during training.
4.   Initiate training of the autoencoder on the given dataset.


```python
# opt = optimizers.Adam(lr= #### FIX ME ####)
# autoencoder_model.compile(optimizer=opt, loss=#### FIX ME ####)
                          
opt = optimizers.Adam(lr= 0.00001)         

autoencoder_model.compile(optimizer=opt, loss="mse")
```


```python
train_history = autoencoder_model.fit(x_train, x_train,
        shuffle= True,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))
```

    Train on 18894 samples, validate on 6298 samples
    Epoch 1/10
    18894/18894 [==============================] - 1s 53us/sample - loss: 0.0612 - val_loss: 0.0543
    Epoch 2/10
    18894/18894 [==============================] - 0s 9us/sample - loss: 0.0599 - val_loss: 0.0528
    Epoch 3/10
    18894/18894 [==============================] - 0s 9us/sample - loss: 0.0587 - val_loss: 0.0514
    Epoch 4/10
    18894/18894 [==============================] - 0s 9us/sample - loss: 0.0575 - val_loss: 0.0500
    Epoch 5/10
    18894/18894 [==============================] - 0s 9us/sample - loss: 0.0563 - val_loss: 0.0486
    Epoch 6/10
    18894/18894 [==============================] - 0s 9us/sample - loss: 0.0551 - val_loss: 0.0473
    Epoch 7/10
    18894/18894 [==============================] - 0s 9us/sample - loss: 0.0540 - val_loss: 0.0459
    Epoch 8/10
    18894/18894 [==============================] - 0s 8us/sample - loss: 0.0529 - val_loss: 0.0447
    Epoch 9/10
    18894/18894 [==============================] - 0s 9us/sample - loss: 0.0518 - val_loss: 0.0435
    Epoch 10/10
    18894/18894 [==============================] - 0s 8us/sample - loss: 0.0508 - val_loss: 0.0423



```python
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.legend(['loss on train data', 'loss on validation data'])
```




    <matplotlib.legend.Legend at 0x7ff2d836a0b8>




![png](output_59_1.png)


## Assessment Task 9: Computing Reconstruction Errors

1.    Fit the trained model on the test dataset. 
2.    Compute the reconstruction scores using MSE as the error metric.


```python
 # Reconstruct the data using our trained autoencoder model
x_test_recon = autoencoder_model.predict(x_test)
# The reconstruction score is the mean of the reconstruction errors (relatively high scores are anomalous)
reconstruction_scores = np.mean((x_test - x_test_recon)**2, axis=1)
```


```python
# Store the reconstruction data in a Pandas dataframe
anomaly_data = pd.DataFrame({'recon_score':reconstruction_scores})

def convert_label_to_binary(labels):
    my_labels = labels.copy()
    my_labels[my_labels != 11] = 1 
    my_labels[my_labels == 11] = 0
    return my_labels
  
# Convert our labels to binary
binary_labels = convert_label_to_binary(y_test)

# Add the binary labels to our anomaly dataframe
anomaly_data['binary_labels'] = binary_labels

# Let's check if the reconstruction statistics are different for labeled anomalies
anomaly_data.groupby(by='binary_labels').describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">recon_score</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>binary_labels</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3328.0</td>
      <td>0.033057</td>
      <td>0.008518</td>
      <td>0.019842</td>
      <td>0.028386</td>
      <td>0.029350</td>
      <td>0.036515</td>
      <td>0.110740</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2970.0</td>
      <td>0.052554</td>
      <td>0.009710</td>
      <td>0.022834</td>
      <td>0.047525</td>
      <td>0.051841</td>
      <td>0.055086</td>
      <td>0.088283</td>
    </tr>
  </tbody>
</table>
</div>



## Assessment Task 10: Anomaly Detection

1.   Plot the area under the curve
2.   Set the optimal threshold that separates normal packets from anomalous packets. 
3.   Threshold should be calculated as the difference between the true positive rate and false positive rate.


```python
fpr, tpr, thresholds = roc_curve(binary_labels, reconstruction_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='lime', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
```


![png](output_66_0.png)



```python
# We can pick the threshold based on the differeence between  the true positive rate (tpr) 
# and the false positive rate (fpr)
optimal_threshold_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_threshold_idx]
print(optimal_threshold)
```

    0.043080635



```python
# Use the optimal threshold value you just printed in the previous cell.
thresh = optimal_threshold

print(thresh)

pred_labels = (reconstruction_scores > thresh).astype(int)

results = confusion_matrix(binary_labels, pred_labels)
# We are using our own accuracy score function in order to grade the assessment
ae_acc = autoencoder_accuracy_score(binary_labels, pred_labels)

# print ('Autoencoder Accuracy Score :', accuracy_score(ae_acc))
print ('Autoencoder Accuracy Score :', autoencoder_accuracy_score(binary_labels, pred_labels))
```

    0.043080635
    Autoencoder Accuracy Score : 0.9217211813274055


In order to pass the assessment, you need to an accuracy of at least 90%.

### Confusion Matrix

This time, we'll create the confusion matrix for you.


```python
print ('Confusion Matrix: ')

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plot_confusion_matrix(results, ['Normal','Anomaly'])
```

    Confusion Matrix: 



![png](output_72_1.png)


## Assessment Task 11: Check Your Assessment Score

Before proceeding, confirm your XGBoost model accuracy is greater than 95% and that your autoencoder accuracy is greater than 90%. If it isn't please continue work on the notebook until you've met these benchmarks.


```python
print ("Accuracy of the XGBoost Model: ", xgb_acc)
print ("Accuracy of the Autoencoder Model: ", ae_acc)
```

    Accuracy of the XGBoost Model:  0.9947602413464592
    Accuracy of the Autoencoder Model:  0.9217211813274055


Run the following cell to grade your assessment.


```python
from assessment import run_assessment
run_assessment()
```

    Testing your XGBoost solution
    Required accuracy greater than 95%....
    Your XGBoost model is accurate enough!
    
    Testing your autoencoder solution
    Required accuracy greater than 90%....
    Your autoencoder model is accurate enough!
    
    You passed the assessment. Congratulations!!!!!
    
    See instructions below for how to get credit for your work.


If the cell above tells you that you passed the assessment, read below for instructions on how to get credit for your work.

### Get Credit for Your Work

To get credit for your assessment and generate a certificate of competency for the course, return to the browser tab where you opened this JupyterLab environment and click the "ASSESS TASK" button, as shown below:

![get_credit](images/get_credit.png)

<a href="https://www.nvidia.com/en-us/deep-learning-ai/education/"> <img src="images/DLI Header.png" alt="Header" style="width: 400px;"/> </a>
