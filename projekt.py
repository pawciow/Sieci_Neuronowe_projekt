#%% [markdown]
# ## Sieci neuronowe projekt 1.
# ## Temat projektu
# Rozpoznawanie jakości wina na podstawie jego właściwośći fizykochemicznych.

#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
data = pd.read_csv('winequality//winequality-white.csv', sep=';')

#%% [markdown]
# ## Wektor danych wejściowych
# Na wejściu mamy wektor 12 wymiarowy. Pierwsze 11 wymiarów to wartości wejściowe, ostatni - 12 - to jest jakość wina czyli jego wartość wyjściowa.
# Wektor danych wygląda następująco:
print(data.head())
#%% [markdown]
# ## Podział danych
# Najpierw oddzielamy dane wejściowe od danych wejściowych, następnie dzielimy dataset
# na dane treningowe i dane testowe
#%%
y = data.quality
X = data.drop('quality',axis=1)

#%% [markdown]
# ## Podział 80%-20%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#%% [markdown]
# Podział 70%-30%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#%% [markdown]
# Podział 90%-10%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

#%% [markdown]
# ## Standaryzacja danych
#%%
sc = preprocessing.StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#%% [markdown] 
# ## Dobór parametrów modelu
#
# Do wybrania funkcji aktywacji użyliśmy kolejno wszystkich możliwych i wybraliśmy funkcję, która 
# daje najlepsze rezultaty.
# 
# Za solver wybraliśmy parametr 'lbfgs' , ponieważ jest on zalecany do mniejszych datasetów
#
#
# TODO: POMYŚLEC NAD alpha : float, optional, default 0.0001
# 
# Parametrów: learning_rate_init, power_t, shuffle, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change nie używaliśmy. 
# Wykluczają się one z naszym solverem. 
#%%
# oszacowanie aktywacji
from sklearn.neural_network import MLPClassifier
activations = ['identity', 'logistic', 'tanh', 'relu']
for item in activations:
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,activation=item,
                               hidden_layer_sizes=(100), random_state=1,
                               )
    classifier.fit(X_train_std,y_train)
    print('Training set score: {} %'.format(classifier.score(X_train_std,y_train)*100))
    print('Training set loss: {} %'.format(classifier.loss_*100))
    # classifier.predict(x)
    # classifier.predict()

