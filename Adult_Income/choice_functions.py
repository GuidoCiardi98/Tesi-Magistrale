import matplotlib.pyplot as plt
import pandas as pd
import xgboost
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate

from useful_functions import *


def preprocess_choice():
    print("\n------------------------------------------------------")
    print("\nINIZIO ESECUZIONE PREPROCESSING:")
    print("\n------------------------------------------------------\n\n")

    # Importo i dati nella variabile data
    data = pd.read_csv("./adult.csv")

    print(data.head())

    # Informazioni generali sul dataset:
    print("Numero di righe del dataset: ", len(data))  # Numero di righe
    print("Numero di colonne del dataset: ", len(data.columns))  # Numero di colonne


    # #########  Per verificare il numero di Missing Values  #########
    data_columns = []

    for i in range(len(data.columns)):
        data_columns.append(data.columns[i])


    for i in data_columns:
        print("Numero di missing values per ", i, " ", check_missing(data[i]))


    # Rimozione delle righe con valori mancanti per gli attributi rilevati prima:
    data['workclass'] = data['workclass'].replace(' ?', np.nan)
    data['occupation'] = data['occupation'].replace(' ?', np.nan)
    data['native-country'] = data['native-country'].replace(' ?', np.nan)

    data = data.dropna(subset=['workclass'])
    data = data.dropna(subset=['occupation'])
    data = data.dropna(subset=['native-country'])


    print("\nRighe nuove: ", len(data))  # Numero di righe

    # Le colonne education e education_num danno la stessa informazione, quindi:
    data = data.drop(["education"], axis=1)

    """# Visualizzazione prima dell'uso di dummies:
    # print("Valori: ", (data['occupation'] == " Armed-Forces").sum())
    attr_distribution = sns.countplot(data=data, x="native-country")
    plt.xticks(fontsize=6)
    plt.show()"""


    # Le variabili numeriche si convertono in dummy:
    data = pd.get_dummies(data)
    data.head()
    print(data)

    # Colonne attuali:
    print(data.columns)
    print(data["Income_ <=50K"])

    # Per specificare la colonna target si mantiene solo la colonna con income>50:
    data = data.drop("Income_ <=50K", axis=1)

    print("Numero di righe del dataset: ", len(data))  # Numero di righe
    print("Numero di colonne del dataset: ", len(data.columns))  # Numero di colonne


    # Inserimento dei dati aggiornati in nuovo file CSV:
    data.to_csv('preprocessed_file.csv', mode='w', index=False, header=True)  # 'a' sta per append;


def visualization_choice():
    print("\n------------------------------------------------------")
    print("\nINIZIO VISUALIZZAZIONE PLOT:")
    print("\n------------------------------------------------------\n\n")

    # Importo i dati nella variabile data
    # data = pd.read_csv("./preprocessed_file.csv")
    data = pd.read_csv("./adult.csv")

    # Parte di visualizzazione dei dati:
    print("Lista delle colonne attuali: \n", data.columns)


    ################################################################
    # VISUALIZZAZIONI DI PARTENZA: #
    """# Generazione plot per una singola colonna:
    attr_distribution = sns.countplot(data=data, x="education")
    plt.xticks(fontsize=7)
    plt.show()
    hs_count = data[(data['education'] == ' Masters') & (data['Income'] == " >50K")].shape[0]
    print("Numero di persone con categoria 'master':", hs_count)"""

    """    # Generazione plot per ciascuna colonna:
    for i in data.columns:
        column_distribution = sns.countplot(data=data, x=i)
        plt.show()"""


    """    # Costruzione pair-plot:
    sns.pairplot(data, height=1)
    plt.show()"""
    ################################################################


    # Confronto tra sex ed Income:
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sex', hue='Income', data=data)

    # Aggiunta di titoli e label agli assi
    plt.title('Relazione tra sex e income')
    plt.xlabel('Sex')
    plt.ylabel('Numero di persone')

    # Mostra il plot
    plt.show()

    # Confronto tra age ed Income:
    plt.figure(figsize=(8, 6))
    sns.countplot(x='age', hue='Income', data=data)

    # Aggiunta di titoli e label agli assi
    plt.title('Relazione tra age e income')
    plt.xlabel('Age')
    plt.xticks(fontsize=8)
    plt.ylabel('Numero di persone')

    # Mostra il plot
    plt.show()


    # Confronto tra education ed Income:
    plt.figure(figsize=(8, 6))
    sns.countplot(x='education', hue='Income', data=data)
    # Aggiunta di titoli e label agli assi
    plt.title('Relazione tra education e income')
    plt.xlabel('Education')
    plt.xticks(fontsize=7)
    plt.ylabel('Numero di persone')

    # Mostra il plot
    plt.show()



def model_management():
    print("\n------------------------------------------------------")
    print("\nINIZIO PARTE GESTIONALE DEI MODELLI:")
    print("\n------------------------------------------------------\n\n")

    data = pd.read_csv("./preprocessed_file.csv")
    np.random.seed(5)
    X = data.drop(columns="Income_ >50K")
    y = data["Income_ >50K"]

    # Splitting dei dati in training e test:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_sub_train, X_validation, y_sub_train, y_validation = train_test_split(X_train, y_train, test_size=0.25)


    print("X Train : ", X_train.shape)
    print("X subsample of train : ", X_sub_train.shape)
    print("X validation : ", X_validation.shape)
    print("X Test  : ", X_test.shape)
    print("Y Train : ", y_train.shape)
    print("Y subsample of train : ", y_sub_train.shape)
    print("Y validation : ", y_validation.shape)
    print("Y Test  : ", y_test.shape)

    # Costruzione modelli
    dec_tree = DecisionTreeClassifier(max_leaf_nodes=100,
                                      max_depth=7)

    adaboost = AdaBoostClassifier(n_estimators=70,
                                  learning_rate=0.7)

    gradient_boosting = GradientBoostingClassifier(n_estimators=5,
                                                   learning_rate=0.5,
                                                   max_depth=6)

    xgboost = XGBClassifier(n_estimators=300,
                            learning_rate=0.7,
                            gamma=0.2,
                            reg_lambda=0.6,
                            max_depth=5,
                            objective='binary:logistic')


    # TRAINING:
    print("\n\n------------------------------------------------------------")
    print("------------------------TRAINING----------------------------")
    print("------------------------------------------------------------\n\n")

    # Addestramento modelli:
    decision_tree_scores = cross_val_score(dec_tree, X_train, y_train, cv=5, scoring='accuracy')
    adaboost_tree_scores = cross_val_score(adaboost, X_train, y_train, cv=5, scoring='accuracy')
    gradient_boosting_scores = cross_val_score(gradient_boosting, X_train, y_train, cv=5, scoring='accuracy')
    # XGBoost possiede in-built cross validation

    dec_tree.fit(X_train, y_train)
    adaboost.fit(X_train, y_train)
    gradient_boosting.fit(X_train, y_train)
    evalset = [(X_validation, y_validation)]
    xgboost.fit(X_train, y_train, early_stopping_rounds=5, eval_metric='logloss', eval_set=evalset)

    # Accuratezze in training:
    print("\nTraining accuracy Decision Tree:", decision_tree_scores.mean())
    print("Training accuracy AdaBoost:", adaboost_tree_scores.mean())
    print("Training accuracy Gradient Boosting:", gradient_boosting_scores.mean())

    y_train_xgb = xgboost.predict(X_train)
    train_score_xgb = accuracy_score(y_train_xgb, y_train)
    print("Training accuracy XGBoost:", train_score_xgb)

    # Calcolo previsioni:
    y_pred_dt = dec_tree.predict(X_test)
    y_pred_ab = adaboost.predict(X_test)
    y_pred_gb = gradient_boosting.predict(X_test)
    y_pred_xgb = xgboost.predict(X_test)

    # Generazione della matrice di confusione:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=dec_tree.classes_)
    sns.heatmap(cm_dt, annot=True, cmap='Blues', fmt='g', ax=axs[0, 0])
    axs[0, 0].set_title('Decision Tree')
    axs[0, 0].set_xlabel('Predicted labels')
    axs[0, 0].set_ylabel('True labels')

    cm_ab = confusion_matrix(y_test, y_pred_ab, labels=adaboost.classes_)
    sns.heatmap(cm_ab, annot=True, cmap='Blues', fmt='g', ax=axs[0, 1])
    axs[0, 1].set_title('Adaboost')
    axs[0, 1].set_xlabel('Predicted labels')
    axs[0, 1].set_ylabel('True labels')

    cm_gb = confusion_matrix(y_test, y_pred_gb, labels=gradient_boosting.classes_)
    sns.heatmap(cm_gb, annot=True, cmap='Blues', fmt='g', ax=axs[1, 0])
    axs[1, 0].set_title('Gradient Boosting')
    axs[1, 0].set_xlabel('Predicted labels')
    axs[1, 0].set_ylabel('True labels')

    cm_xgb = confusion_matrix(y_test, y_pred_xgb, labels=xgboost.classes_)
    sns.heatmap(cm_xgb, annot=True, cmap='Blues', fmt='g', ax=axs[1, 1])
    axs[1, 1].set_title('XGBoost')
    axs[1, 1].set_xlabel('Predicted labels')
    axs[1, 1].set_ylabel('True labels')

    # disp.plot()
    plt.tight_layout()
    # plt.show()

    # Risultati di accuratezza:
    score_dt = accuracy_score(y_test, y_pred_dt)
    print('\nTest accuracy Decision Tree: %.3f' % score_dt)

    score_adab = accuracy_score(y_test, y_pred_ab)
    print('Test accuracy AdaBoost: %.3f' % score_adab)

    score_gb = accuracy_score(y_test, y_pred_gb)
    print('Test accuracy Gradient Boosting: %.3f' % score_gb)

    score_xgb = accuracy_score(y_test, y_pred_xgb)
    print('Test accuracy XGBoost: %.3f' % score_xgb)


