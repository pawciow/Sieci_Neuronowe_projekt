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
# Jako pierwszy model zaimplementuję sieć używającą DecisionTreeClasifier

#%%
# DecisionTreeClasifier
from sklearn import tree
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)
# Sprawdzanie dokładności
confidence = classifier.score(X_test, y_test)

#%% [markdown]
# ## Wyniki dla drzewa decyzyjnego#
print('Wyniki dla klasyfikatora drzewa decyzyjnego:{}'.format(confidence*100))

#%% [markdown]
# ## Wyniki z poprzednich prób:
#
# Dla podziału 70%-30%:
# 56.46%, 56.53%, 56.25%, 59.25%
#
# Dla podziału 80%-20%:
# 60.00%, 59.08%, 59.79%, 61.93%
#
# Dla podziału 90%-10%:
# 56.12%, 56.12%, 58.97%, 64.69%
#
#%%
# ## Graf:
import graphviz 
dot_data = tree.export_graphviz(classifier, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("DecisionTreeClasifier")
dot_data = tree.export_graphviz(classifier, out_file=None, 
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph
