# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import visuals as vs
data = pd.read_csv('census.csv')

income_raw = data['income']
features_raw = data.drop('income',axis = 1)

sns.distplot(data['capital-loss'])
sns.distplot(data['education-num'])


  vs.distribution(data)  

skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))


vs.distribution(features_log_transformed, transformed = True)


%%normalization

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical = ['age','education-num','capital-gain','capital-loss','hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)

features_log_minmax_transform[numerical]  = scaler.fit_transform(features_log_transformed[numerical])


features_final = pd.get_dummies(features_log_minmax_transform)

income = income_raw.apply(lambda x: 1 if x=='>50K' else 0)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(features_final,income,test_size = 0.2,random_state = 0)

from sklearn.metrics import fbeta_score,accuracy_score




from sklearn.metrics import fbeta_score,accuracy_score
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

results={}

results['accuracy'] = accuracy_score(y_test,y_pred)

results['f-score'] = fbeta_score(y_test,y_pred,beta = 0.5)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

clf = AdaBoostClassifier(random_state = 1)

parameters = {'n_estimators' : [50, 100, 125, 150],
              'learning_rate' : [0.1, 0.5, 1.0, 1.5]
                   }
    
scorer= make_scorer(fbeta_score,beta = 0.5)
grid_search = GridSearchCV(clf,parameters,scoring = scorer)
grid_search =  grid_search.fit(X_train,y_train)

best_parameters = grid_search.best_params_

best_f1score = grid_search.best_score_


















