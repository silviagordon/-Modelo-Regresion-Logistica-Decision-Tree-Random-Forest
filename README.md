# -Modelo-Regresion-Logistica-Decision-Tree-Random-Forest
Proyecto académico de regresión logística utilizando las técnicas Decision Tree y Random Forest para conocer la fiabilidad del modelo.

En este repositorio se encuentran divididos por diferentes carpetas el proceso de análisis exploratorio, preprocesado y comparación de métricas de los modelos Decision Tree y Randon Forest para un conjuntos de datos que consiste en las calificaciones obtenidas por los estudiantes en varias materias en EEUU.

Dicho repositorio se divide en:

.- Carpeta Regresión Logística:
    
    - 01 - Regresion-Logistica-EDA.

    - 02 - Regresion-Logistica-Preprocesado.

    - 03 - Regresion-Logistica-Intro.

    - 04 - Regresion-Logistica-Metricas.

    - 05 - Regresion-Logistica-Decision-Tree.

    - 06 - Regresion-Logistica-Random-Forest.

Las librerías utilizadas en este repositorio han sido:

### Tratamiento de datos
------------------------------------------------------------------------------

import numpy as np

import pandas as pd

from tqdm import tqdm

### Gráficos
------------------------------------------------------------------------------

import matplotlib.pyplot as plt

import seaborn as sns

### Modelado y evaluación
------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score , 
cohen_kappa_score, roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

###  Crossvalidation
------------------------------------------------------------------------------

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn import metrics

### Estandarización variables numéricas y Codificación variables categóricas
------------------------------------------------------------------------------

from scipy import stats

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

import math

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder # para realizar el Label Encoding 

from sklearn.preprocessing import OneHotEncoder  # para realizar el One-Hot Encoding

### Estadísticos
------------------------------------------------------------------------------

import statsmodels.api as sm

from statsmodels.formula.api import ols

import researchpy as rp

from scipy.stats import skew

from scipy.stats import kurtosistest

### Gestión datos desbalanceados
------------------------------------------------------------------------------

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler

from imblearn.combine import SMOTETomek

### Configuración warnings
------------------------------------------------------------------------------

import warnings

warnings.filterwarnings('ignore')

### Establecer tamaño gráficas
------------------------------------------------------------------------------

plt.rcParams["figure.figsize"] = (15,15)