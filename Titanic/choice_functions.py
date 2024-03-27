import matplotlib.pyplot as plt
import pandas as pd
import xgboost
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate

from useful_functions import *


def preprocess_choice():
    print("\n------------------------------------------------------")
    print("\nINIZIO ESECUZIONE PREPROCESSING:")
    print("\n------------------------------------------------------\n\n")

    # Importo i dati nella variabile data
    data = pd.read_csv("./titanic_dataset.csv")
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
    # #################################################################################

    # Si rinominano i valori di 'Sex' da F,M rispettivamente a 0,1:
    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] == 'female', 'Sex'] = 1

    # Lo stesso si fa per 'Embarked' passando da S, C, Q rispettivamente ad 1, 2, 3:
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 2
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 3


    # Rimozione delle colonne ritenute inutili ai fini dell'analisi:
    data.drop(['Name', 'Ticket', 'Fare', 'Cabin'], inplace=True, axis=1)

    # Rimozione delle righe con valori mancanti per gli attributi rilevati prima:
    data.dropna(subset=['Age'], inplace=True)
    data.dropna(subset=['Embarked'], inplace=True)

    print("Numero di righe del dataset: ", len(data))  # Numero di righe
    print("Numero di colonne del dataset: ", len(data.columns))  # Numero di colonne


    # Inserimento dei dati aggiornati in nuovo file CSV:
    data.to_csv('preprocessed_file.csv', mode='w', index=False, header=True)  # 'a' sta per append;

    # La suddivisione dei dati in training set e test set è stata svolta a partire dal nuovo file preprocessato :'preprocessed_file.csv'.


def visualization_choice():
    print("\n------------------------------------------------------")
    print("\nINIZIO VISUALIZZAZIONE PLOT:")
    print("\n------------------------------------------------------\n\n")

    # Importo i dati nella variabile data
    data = pd.read_csv("./preprocessed_file.csv")

    # Parte di visualizzazione dei dati:
    print("Lista delle colonne attuali: \n", data.columns)



    ################################################################
    # VISUALIZZAZIONI DI PARTENZA: #
    """# Generazione plot per una singola colonna:
    attr_distribution = sns.countplot(data=data, x="Embarked")
    plt.xticks(fontsize=8)
    plt.show()"""

    """    # Generazione plot per ciascuna colonna:
    for i in data.columns:
        column_distribution = sns.countplot(data=data, x=i)
        plt.show()"""


    """    # Costruzione pair-plot:
    sns.pairplot(data, height=1)
    plt.show()"""
    ################################################################

    # Percentuale di sopravvivenza dei dati forniti:
    titanic_survived = data[data["Survived"] == 1]
    print(data.groupby("Survived")["PassengerId"].count())
    print(data.groupby("Sex")["PassengerId"].count())
    print("\nPercentuale di sopravvissuti: ", len(titanic_survived) / len(data) * 100, "%\n")


    # Grafico per il tasso di sopravvivenza dei passeggeri:
    titanic_sex = data.groupby(["Survived", "Sex"]).count()["PassengerId"]
    # print(titanic_sex)
    # print("T0: ", titanic_sex[0])
    # print("T1: ", titanic_sex[1])
    (titanic_sex[1] / (titanic_sex[1] + titanic_sex[0])).plot(kind="pie", autopct='%1.1f%%', figsize=(5, 5))
    plt.legend(["Male=0", "Female=1"], loc="lower right")
    plt.xlabel("Survival rate per Sex")
    # print("Surval rate based on", titanic_sex[1] / (titanic_sex[1] + titanic_sex[0]))
    plt.show()


    # Grafico per il tasso di sopravvivenza in base a Sex per ciascuna classe:
    titanic_sex = data.groupby(["Survived", "Sex"])["Sex"].count()
    data.groupby("Sex").count()["PassengerId"].plot(kind="barh", color="g", width=0.3, legend=True, label="Total")
    titanic_sex[0].plot(kind="barh", color="r", legend=True, position=0, width=0.25, label="Not Survived")
    titanic_sex[1].plot(kind="barh", color="b", legend=True, position=1, width=0.25, label="Survived")
    plt.xlabel("Numero di passeggeri a bordo")
    plt.ylabel("Sex")
    plt.show()


    # Percentuale di sopravvivenza:
    titanic_sex_percentage = data.groupby(["Survived", "Sex"])["Sex"].count() / data["Sex"].count()
    # print(titanic_sex_percentage)


    # Studio del tasso di sopravvivenza in base a Pclass:
    titanic_pclass = data.groupby(["Survived", "Pclass"])["Pclass"].count()

    (titanic_pclass[1] / (titanic_pclass[0] + titanic_pclass[1])).plot(kind="pie", autopct='%1.1f%%', figsize=(7, 5))
    plt.xlabel("Survival rate per Pclass")
    plt.legend(["Prima classe=1", "Seconda classe=2", "Terza classe=3"], bbox_to_anchor=(1.1, 1.1),  loc="upper right")
    plt.show()


    # Tasso di sopravvivenza per Pclass in ciascuna classe:
    data.groupby("Pclass").count()["PassengerId"].plot(kind="bar", color="g", width=0.3, legend=True, label="Total")
    titanic_pclass[0].plot(kind="bar", color="r", legend=True, position=0, width=0.25, label="Not Survived")
    titanic_pclass[1].plot(kind="bar", color="b", legend=True, position=1, width=0.25, label="Survived")
    plt.xlabel("Passenger class")
    plt.ylabel("Numero di passeggeri a bordo")
    plt.show()
    # print(titanic_pclass)
    # print(titanic_pclass_percentage)


    # Studio del tasso di sopravvivenza in base a Sex e Pclass:
    titanic_sex_pclass = data.groupby(["Survived", "Sex", "Pclass"])["Pclass"].count()

    (titanic_sex_pclass[1] / (titanic_sex_pclass[0] + titanic_sex_pclass[1])).plot(kind="pie", autopct='%1.1f%%', figsize=(7, 5))
    plt.xlabel("(Sex,Passenger Class)")
    plt.ylabel("Tasso di sopravvivenza")
    plt.show()

    # Tasso di sopravvivenza per Pclass in ciascuna classe:
    (titanic_sex_pclass[0] + titanic_sex_pclass[1]).plot(kind="bar", color="g", legend=True, width=0.3, label="Total")
    titanic_sex_pclass[0].plot(kind="bar", color="r", legend=True, position=0, width=0.25, label="Not Survived")
    titanic_sex_pclass[1].plot(kind="bar", color="b", legend=True, position=1, width=0.25, label="Survived")
    plt.xlabel("(Sex,Passenger Class)")
    plt.ylabel("Numero di passeggeri a bordo")
    plt.show()
    # print(titanic_sex_pclass)
    # print(titanic_sex_pclass_percentage)


    # Studio della sopravvivenza in base a Sex e Age:
    g = sns.FacetGrid(data, col="Sex", row="Survived", margin_titles=True)
    g.map(sns.histplot, "Age", bins=100, color="royalblue")

    plt.show()

    #######################################################
    # Passeggeri minorenni:
    """age_less_18 = data[data['Age'] < 18].shape[0]
    print("Numero di passeggeri minorenni: ", age_less_18)
    pd.set_option('display.max_rows', None)
    print(data.groupby(["Survived", "Sex", "Age"])["Age"].count())
    pd.reset_option('display.max_rows')"""
    #######################################################


    # Tasso di sopravvivenza per di Embarked:
    titanic_embarked = data.groupby(["Survived", "Embarked"]).count()["PassengerId"]

    (titanic_embarked[1] / (titanic_embarked[0] + titanic_embarked[1])).plot(kind="pie", autopct='%1.1f%%', figsize=(7, 5))
    plt.xlabel("Embarked Port")
    plt.ylabel("Tasso di sopravvivenza")
    plt.legend(["C = Cherbourg", "Q = Queenstown", "S = Southampton"], bbox_to_anchor=(1.1, 1.1),  loc="upper right")
    plt.show()


    # Tasso di sopravvivenza per Embarked in ciascuna classe:
    titanic_embarked = data.groupby(["Survived", "Embarked"])["Embarked"].count()
    data.groupby("Embarked").count()["PassengerId"].plot(kind="bar", color="g", width=0.3, legend=True, label="Total")
    titanic_embarked[0].plot(kind="bar", color="r", legend=True, position=0, width=0.25, label="Not Survived")
    titanic_embarked[1].plot(kind="bar", color="b", legend=True, position=1, width=0.25, label="Survived")
    plt.ylabel("Numero di passeggeri a bordo")
    plt.xlabel("Embarked Port")
    plt.show()



    # CONFRONTO PARCH E SIBSP:

    # Tasso di sopravvivenza per Parch:
    titanic_Parch = data.groupby(["Survived", "Parch"])["Parch"].count()
    titanic_Parch_percentage = data.groupby(["Survived", "Parch"])["Parch"].count() / len(data)
    print(titanic_Parch)
    print(titanic_Parch_percentage)

    plt.subplot(131)
    (titanic_Parch[0] + titanic_Parch[1]).plot(kind="bar", color="g", legend=True, width=0.3, label="Total")
    titanic_Parch[0].plot(kind="bar", color="r", legend=True, position=0, width=0.25, label="Not Survived")
    titanic_Parch[1].plot(kind="bar", color="b", legend=True, position=1, width=0.25, label="Survived")
    plt.xlabel("Numero di passeggeri in viaggio con genitori/figli")
    plt.ylabel("Numero di passeggeri")


    # Tasso di sopravvivenza per SibSp:
    titanic_SibSp = data.groupby(["Survived", "SibSp"])["SibSp"].count()
    titanic_SibSp_percentage = data.groupby(["Survived", "SibSp"])["SibSp"].count() / len(data)
    print(titanic_SibSp)
    print(titanic_SibSp_percentage)

    plt.subplot(132)
    (titanic_SibSp[0] + titanic_SibSp[1]).plot(kind="bar", color="g", legend=True, width=0.3, label="Total")
    titanic_SibSp[0].plot(kind="bar", color="r", legend=True, position=0, width=0.25, label="Not Survived")
    titanic_SibSp[1].plot(kind="bar", color="b", legend=True, position=1, width=0.25, label="Survived")
    plt.xlabel("Numero di passeggeri in viaggio con fratelli/sposi")
    plt.ylabel("Numero di passeggeri")


    # Tasso di sopravvivenza per Parch e SibSp:
    titanic_SibSp_Parch = data.groupby(["Survived", "Parch", "SibSp"])["Parch"].count()
    titanic_SibSp_Parch_percentage = data.groupby(["Survived", "Parch", "SibSp"])["Parch"].count() / len(data)
    print(titanic_SibSp_Parch)
    print(titanic_SibSp_Parch_percentage)

    plt.subplot(133)
    (titanic_SibSp_Parch[0] + titanic_SibSp_Parch[1]).plot(kind="bar", color="g", legend=True, width=0.55, label="Total")
    titanic_SibSp_Parch[0].plot(kind="bar", color="r", legend=True, position=0, width=0.45, label="Not Survived")
    titanic_SibSp_Parch[1].plot(kind="bar", color="b", legend=True, position=1, width=0.45, label="Survived")
    plt.xlabel("Numero di passeggeri in viaggio con (genitori/figli, fratelli/sposi)")
    plt.ylabel("Numero di passeggeri")

    plt.show()


    # In percentuale per Parch:
    plt.subplot(131)
    (titanic_Parch_percentage[0] + titanic_Parch_percentage[1]).plot(kind="bar", color="g", legend=True, width=0.3,
                                                                     label="Total")
    titanic_Parch_percentage[0].plot(kind="bar", color="r", legend=True, position=0, width=0.25, label="Not Survived")
    titanic_Parch_percentage[1].plot(kind="bar", color="b", legend=True, position=1, width=0.25, label="Survived")
    plt.xlabel("Numero di passeggeri in viaggio con genitori/figli")
    plt.ylabel("Numero di passeggeri")

    # In percentuale per SibSp:
    plt.subplot(132)
    (titanic_SibSp_percentage[0] + titanic_SibSp_percentage[1]).plot(kind="bar", color="g", legend=True, width=0.3,
                                                                     label="Total")
    titanic_SibSp_percentage[0].plot(kind="bar", color="r", legend=True, position=0, width=0.25, label="Not Survived")
    titanic_SibSp_percentage[1].plot(kind="bar", color="b", legend=True, position=1, width=0.25, label="Survived")
    plt.xlabel("Numero di passeggeri in viaggio con fratelli/sposi")
    plt.ylabel("Numero di passeggeri")


    # In percentuale per Parch e SibSp:
    plt.subplot(133)
    (titanic_SibSp_Parch_percentage[0] + titanic_SibSp_Parch_percentage[1]).plot(kind="bar", color="g", legend=True, width=0.55,
                                                                   label="Total")
    titanic_SibSp_Parch_percentage[0].plot(kind="bar", color="r", legend=True, position=0, width=0.45, label="Not Survived")
    titanic_SibSp_Parch_percentage[1].plot(kind="bar", color="b", legend=True, position=1, width=0.45, label="Survived")
    plt.xlabel("Numero di passeggeri in viaggio con (genitori/figli, fratelli/sposi)")
    plt.ylabel("Numero di passeggeri")

    plt.show()


def model_management():
    print("\n------------------------------------------------------")
    print("\nINIZIO PARTE GESTIONALE DEI MODELLI:")
    print("\n------------------------------------------------------\n\n")

    data = pd.read_csv("./preprocessed_file.csv")
    np.random.seed(5)  #
    X = data.drop(columns="Survived")
    y = data["Survived"]

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
    dec_tree = DecisionTreeClassifier(max_depth=5,
                                      max_leaf_nodes=10)

    adaboost = AdaBoostClassifier(n_estimators=300,
                                  learning_rate=0.05)

    gradient_boosting = GradientBoostingClassifier(n_estimators=219,
                                                   learning_rate=0.6,
                                                   max_depth=4)
    xgboost = XGBClassifier(n_estimators=30,
                            learning_rate=0.9,
                            gamma=0.7,
                            reg_lambda=50,
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
    gradient_boosting.fit(X_train, y_train)
    adaboost.fit(X_train, y_train)
    evalset = [(X_validation, y_validation)]
    xgboost.fit(X_train, y_train, early_stopping_rounds=10, eval_metric='logloss', eval_set=evalset)


    # Accuratezze in training:
    print("\nDecision Tree Accuracy:", decision_tree_scores.mean())
    print("AdaBoost Accuracy:", adaboost_tree_scores.mean())
    print("Gradient Boosting Accuracy:", gradient_boosting_scores.mean())

    y_train_xgb = xgboost.predict(X_train)
    train_score_xgb = accuracy_score(y_train_xgb, y_train)
    print("Training accuracy XGBoost:", train_score_xgb)

    # Calcolo previsioni:
    y_pred_dt = dec_tree.predict(X_test)
    y_pred_ab = adaboost.predict(X_test)
    y_pred_gb = gradient_boosting.predict(X_test)
    y_pred_xgb = xgboost.predict(X_test)


    # Generazione delle matrici di confusione:
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
    plt.show()


    # Risultati di accuratezza:
    score_dt = accuracy_score(y_test, y_pred_dt)
    print('\nTest accuracy Decision Tree: %.3f' % score_dt)

    score_adab = accuracy_score(y_test, y_pred_ab)
    print('Test accuracy AdaBoost: %.3f' % score_adab)

    score_gb = accuracy_score(y_test, y_pred_gb)
    print('Test accuracy Gradient Boosting: %.3f' % score_gb)

    score_xgb = accuracy_score(y_test, y_pred_xgb)
    print('Test accuracy XGBoost: %.3f' % score_xgb)