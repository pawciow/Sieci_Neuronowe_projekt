#%% [markdown]
# ## Sieci neuronowe projekt 1.
# ## Temat projektu
# Rozpoznawanie jakości wina na podstawie jego właściwośći fizykochemicznych.

#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import load_wine # todo: THIS
data = pd.read_csv('winequality//winequality-red.csv', sep=';')

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
# Za solver wybraliśmy parametr 'adam' , ponieważ jest on zalecany do mniejszych datasetów
#
#  dla 'lbfgs' otrzymywaliśmy wyniki na poziomie 65%
#
# TODO: POMYŚLEC NAD alpha : float, optional, default 0.0001
# 
# Parametrów: learning_rate_init, power_t, shuffle, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change nie używaliśmy. 
# Wykluczają się one z naszym solverem. 
#%%
# oszacowanie aktywacji
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# activations = ['identity', 'logistic', 'tanh', 'relu']
activations = ['tanh','relu']
params, scores = [],[]
for a in range(-5,6):
    classifier = MLPClassifier(solver='adam', alpha=10**a,activation='relu',
                               hidden_layer_sizes=(50,25,10,5), random_state=1,
                               max_iter=1000)
    classifier.fit(X_train_std,y_train)
    print('Training set score: {} %'.format(classifier.score(X_train_std,y_train)*100))
    print('Training set loss: {} %'.format(classifier.loss_*100))
    print('Test accuracy: {}%'.format(classifier.score(X_test_std,y_test)*100))
    # yPredMLP = classifier.predict(X_test_std)
    # print('NEWTESTING: {}'.format(yPredMLP))
    classifier.fit(X_train_std, y_train)
    yPredMLP = classifier.predict(X_test_std)
    score = accuracy_score(y_test,yPredMLP)
    params.append(classifier.alpha)
    scores.append(score)
plt.plot(params,scores,linestyle='--',color='blue', label='MLP')
plt.legend(loc='lower right')
plt.xscale('log')
plt.xlabel('parameter values')
plt.ylabel('Accuracy')
plt.title('Accuracy scores')
plt.show()

    # classifier.predict(x)
    # classifier.predict()

