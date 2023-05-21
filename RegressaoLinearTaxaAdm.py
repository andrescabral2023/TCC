#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Importando as bibliotecas

import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

# Importando a base de dados

dataset = pd.read_csv("/Users/andre/downloads/MBA/dados_TCC/CAD_FI_SANEADO.csv", encoding='Latin-1', sep=",")
X = dataset.iloc[:,[False, False, True, False, True, True, False, False, True, True, False, False, False, False, False, True]]
y = dataset.iloc[:, -6].values

# Variáveis "dummy"

X_dummies = pd.get_dummies(X)

# Dados de treino e de teste

X_train, X_test, y_train, y_test = ms.train_test_split(X_dummies, y, test_size = 0.2, random_state = 0)

# Treinando o modelo

#%% REGRESSÃO RIDGE

ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
test_predict = ridge_model.predict(X_test)
train_predict = ridge_model.predict(X_train)
ridge_model.score(X_test, y_test)

from sklearn.metrics import mean_squared_error
train_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
test_rmse = np.sqrt(mean_squared_error(y_train, train_predict))

# Validação cruzada

num_folds = 30
seed = 7

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

modelo = Ridge()
resultado = cross_val_score(modelo, X_dummies, y, cv = kfold)
resultado.mean()
#%% REGRESSÃO LASSO

#from sklearn.linear_model import Lasso
#lasso_model = Lasso()
#lasso_model.fit(X_train, y_train)

#test_predict = lasso_model.predict(X_test)
#train_predict = lasso_model.predict(X_train)
#lasso_model.score(X_test, y_test)
#from sklearn.metrics import mean_squared_error
#train_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
#test_rmse = np.sqrt(mean_squared_error(y_train, train_predict))

#%% REGRESSAO LINEAR

#regressor = lm.LinearRegression()
#regressor.fit(X_train, y_train)
#regressor.score(X_test, y_test)

#%%
# Previsão

y_pred = ridge_model.predict(X_test)

np.set_printoptions(precision=2)
result = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)

# 7. Juntando a previsão à tabela

def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

X_reverse = undummify(X_test)

X_reverse = X_reverse.reset_index(drop=True)

y_compare = pd.DataFrame(result)
y_compare = y_compare.rename(index=str, columns={0:'y_pred', 1:'y_test'})
y_compare = y_compare.reset_index(drop=True)

# Resultado final

resultado_final = pd.concat([y_compare, X_reverse], axis=1)

