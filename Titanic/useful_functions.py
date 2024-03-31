import numpy as np
import pandas as pd  # Per Data Processing, I/O CSV file (pd.read_csv)
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Funzione per la ricerca di missing values:
def check_missing(column):
    n_missing = 0

    for i in range(len(column)):
        if pd.isnull(column.values[i]):
            n_missing = n_missing + 1

    return n_missing
