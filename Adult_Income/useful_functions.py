import numpy as np
import pandas as pd  # Per Data Processing, I/O CSV file (pd.read_csv)
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Funzione per la ricerca di missing values:
def check_missing(column):
    n_missing = 0

    for i in range(len(column)):
        if pd.isnull(column.values[i]) or column.values[i] == ' ?':
            n_missing = n_missing + 1

    return n_missing


# Funzione per la normalizzazione dell'intervallo [0,1]:
def min_max_normalization(data_column):
    actual = data_column
    min_value = data_column.min()
    max_value = data_column.max()
    return (actual - min_value) / (max_value - min_value)