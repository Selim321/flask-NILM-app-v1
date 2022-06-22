import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import os
abs_path = os.getcwd()
fr = pd.read_csv(abs_path +"\\fridge.csv")
wm = pd.read_csv(abs_path +"\\WM.csv")


# converting to arrays
fr_arr = fr.to_numpy()
wm_arr = wm.to_numpy()

#labeling
thr = 20
y_fr=[]
for i in fr_arr:
    if i>=thr:
        y_fr.append(1)
    else:
        y_fr.append(0)

y_wm=[]
for i in wm_arr:
    if i>thr:
        y_wm.append(1)
    else:
        y_wm.append(0)

y_fr_arr = np.array(y_fr)
y_fr_arr = y_fr_arr.reshape(len(y_fr_arr), 1)

y_wm_arr = np.array(y_wm)
y_wm_arr = y_wm_arr.reshape(len(y_wm_arr), 1)

# label matrix
Y= np.concatenate((y_fr_arr, y_wm_arr), axis=1)

# feature matrix
X = fr_arr + wm_arr

# split train/test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, shuffle=False)

# import model 
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain 
#%%

""" classifier """   

chains = [ClassifierChain(RandomForestClassifier(random_state=0), order='random', random_state=i)
  for i in range(10)]
for chain in chains:
    chain.fit(X_train, y_train)

Y_pred_chains = np.array([chain.predict(X_train) for chain in
                  chains])
Y_pred_train = Y_pred_chains.mean(axis=0)

Y_pred_chains = np.array([chain.predict(X_test) for chain in
          chains])
Y_pred_test = Y_pred_chains.mean(axis=0)

Y_pred_chains = np.array([chain.predict(X) for chain in
          chains])
Y_pred_all = Y_pred_chains.mean(axis=0)


#%%

""" 
 * the use give a time interval of length n as input
 ** the app returns a (2, n) vector of length n corresponding to the states of each laod
""" 
user_input = [5,7]

app_output = np.array([chain.predict(X[user_input[0] : user_input[1]]) for chain in
          chains])
app_output = app_output.mean(axis=0)

#%%
""" model persistance """
from joblib import dump, load
#dump(chains, 'my_model.joblib') 

""" data persistance """
#dump(X, 'data.joblib')
