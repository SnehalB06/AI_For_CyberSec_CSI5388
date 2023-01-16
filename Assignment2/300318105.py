'''import the needed modules here:'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

pass

'''
In order to have reproducible results, you need to set the 
random states (a.k.a seeds) of the sources of randomness to
a constant number. You can set the local seeds to a constant
number in the respective place. If you have global seeds, 
set them here:
'''
pass


def train_then_test(x_train, y_train, x_test):
    '''
    This functions has the following inputs:
     * x_train: which is a numpy list of the training data
     * y_train: which is a numpy list of the corresponding labels for the given features
     * x_test: which is a numpy list test data


    Within the body of this function you need to build and train a
    classifier for the given training data (x_train, y_train) and
    then return the predicted labels for x_test.
    Output: predicted_labels (can be a python list, numpy list, or pandas list of the predicted labels)

    Notes:
        Do not change the name of this function.
        Do not add new parameters to this function.
        Do not remove the parameters of this function (x_train, y_train, x_test)
        Do not change the order of the parameters of this function
        If x_test contains 100 rows of data, predicted_labels needs to have 100 rows of predicted labels
    '''
    
    
    X_Train_data = x_train
    X_Test_data = x_test
    Y_Train_data = y_train
    
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_Train_data)
    X_test_std = scaler.fit_transform(X_Test_data)
    
    clf = AdaBoostClassifier(n_estimators=1000, learning_rate=1.0, random_state=10, algorithm='SAMME')
    clf.fit(X_train_std, Y_Train_data.values.ravel())
    clf_adaboost_predictions=clf.predict(X_test_std)
    clf_adaboost_predictions = pd.DataFrame(clf_adaboost_predictions, columns=['Predicted'])

    return clf_adaboost_predictions

    '''
    d = preprocessing.normalize(X_Test_data)
    Normalise_X_Test_data = pd.DataFrame(d)
    Normalise_X_Test_data.head()

    d = preprocessing.normalize(X_Train_data)
    Normalise_X_Train_data = pd.DataFrame(d)
    Normalise_X_Train_data.head()
    
    clf_DT= DecisionTreeClassifier()
    clf = AdaBoostClassifier(n_estimators=1000,base_estimator=clf_DT, learning_rate=1.0, random_state=10, algorithm='SAMME')
    clf.fit(Normalise_X_Train_data, Y_Train_data.values.ravel())

    clf_adaboost_predictions=clf.predict(Normalise_X_Test_data)

    '''
    
    
    # return trained_model.predict(x_test)
    pass


if __name__ == '__main__':
    # You need to do all the function calls for testing purposes in this scope:
    X_Train_data = pd.read_csv(r"E:\uOttawaTerm1\AI For CyberSec\Assign2\x_train.csv")
    X_Train_data.astype(float)
    print(X_Train_data)
    
    Y_Train_data = pd.read_csv(r"E:\uOttawaTerm1\AI For CyberSec\Assign2\y_train.csv")
    Y_Train_data.astype(int)
    print(Y_Train_data)

    X_Test_data = pd.read_csv(r"E:\uOttawaTerm1\AI For CyberSec\Assign2\x_test.csv")
    X_Test_data.astype('float')
    print(X_Test_data)

    
    model = train_then_test(X_Train_data,Y_Train_data,X_Test_data)
    print(model)
    
    pass
