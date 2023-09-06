# imports

from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import pandas as pd
import time
import os
import csv


def KFold_Validation(X, y, nSplits=5):
    # Initialize the K-fold cross-validator
    kf = KFold(n_splits=nSplits, shuffle=True, random_state=42)

    Testing_Results = {}

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        # Split the data into training and testing sets for this fold
        print("K-Fold: ", i)
        X_t, X_test = X.iloc[train_index], X.iloc[test_index]
        y_t, y_test = y.iloc[train_index], y.iloc[test_index]

        smo = SMOTE(random_state=42)
        X_train, y_train = smo.fit_resample(X_t, y_t)

        # Train the model
        model = train(X_train, y_train)

        # Test the model
        precision, recall, f1 = test(X_test, y_test, model)

        # Store the results
        Testing_Results[i] = (precision, recall, f1)

    return Testing_Results


'''
    Change the train method according to classifier used
'''
def train(X_train, y_train):
    """
    Use grid search to find optimal parameters and then test the optimal model
    :param X_train:
    :param y_train:
    :return: None
    """
    # grid search parameter list
    tuned_parameters = {
        'splitter': ('best', 'random'),
        'criterion': ("gini", "entropy"),
        "max_depth": [*range(20, 51, 10)],
        'min_samples_leaf': [*range(1, 15, 2)]
    }
    # 生成模型
    print("Start training : " + "\n")
    grid = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='roc_auc', verbose=2, n_jobs=4)
    # Send data to model training
    model = grid.fit(X_train, y_train)
    return model


def test(X_test, y_test, estimator):
    y_pre = estimator.predict(X_test)
    precision = precision_score(y_test, y_pre, labels=None, pos_label=1, average='binary', sample_weight=None)
    recall = recall_score(y_test, y_pre, labels=None, pos_label=1, average='binary', sample_weight=None)
    f1 = f1_score(y_test, y_pre, labels=None, pos_label=1, average='binary', sample_weight=None)

    # precision = precision_score(y_test, y_pre, labels=None, average='macro', sample_weight=None)
    # recall = recall_score(y_test, y_pre, labels=None, average='macro', sample_weight=None)
    # f1 = f1_score(y_test, y_pre, labels=None, average='macro', sample_weight=None)

    print("precision: {:.3}, recall: {:.3}, f1:{:.3}, \n".format(precision, recall, f1))

    return precision, recall, f1


''' 
    Change the input path accordingly
'''
train_dir = r"L:\rems_modified\Training_CSV"

Final_Results = {}
result_index = 0

for Code_Emb in os.listdir(train_dir):
    print(Code_Emb)
    Final_Results[Code_Emb] = {}

    for Tree_Emb in os.listdir(train_dir + "\\" + Code_Emb):
        print("\t" + Tree_Emb.split(".")[0])

        curr_dir = train_dir + "\\" + Code_Emb + "\\" + Tree_Emb

        data = pd.read_csv(curr_dir, header=None)

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Get the current time
        start_time = time.time()

        # Execute some code you want to time
        Results = KFold_Validation(X, y, nSplits=10)

        # Get the elapsed time
        elapsed_time = round((time.time() - start_time) / 60, 3)

        print("Elapsed time: ", elapsed_time, " Minutes")

        results_ = pd.DataFrame(Results, index=["Precision", "Recall", "F1"]).T
        mean_results = dict(results_.describe().loc["mean"])

        out = [result_index, Code_Emb, Tree_Emb.split(".")[0], mean_results['Precision'], mean_results['Recall'], mean_results['F1']]
        print(result_index, Code_Emb, Tree_Emb.split(".")[0], mean_results['Precision'], mean_results['Recall'], mean_results['F1'])
        with open("DT_output.csv", 'a+', newline='') as out_csv:
            csv_writer = csv.writer(out_csv)
            csv_writer.writerow(out)
        result_index += 1
