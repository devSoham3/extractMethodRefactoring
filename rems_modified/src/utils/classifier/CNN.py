# -*- encoding: utf-8 -*-
"""
@description: 用CNN训练并对REMS数据进行测试
@date: 2022/10/1
@File : CNN.py
@Software : PyCharm
"""
import datetime
import numpy as np
import pandas as pd
import os
import sys
import platform
import csv
import tensorflow as tf
from sklearn import metrics
import keras_metrics as km
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

_training_data_path = sys.argv[1]  # Training dataset file path
_REMS_project_path = sys.argv[2]  # Test dataset file path


def load_data(src):
    """
    Load the training dataset and split the training and test sets
    :param src: training set file
    :return: vec and label of the training set, vector length
    """
    df = pd.read_csv(src, header=None)
    X = []  # 向量
    y = []  # 标签
    lens = 0
    with open(src) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            a = row[0:-1]
            lens = len(a)
            b = []
            for x in a:
                b.append(float(x))
            X.append(b)
            if row[-1].upper() == "TRUE":
                y.append(1)
            elif row[-1].upper() == "FALSE":
                y.append(0)
    smo = SMOTE(random_state=42)
    X, y = smo.fit_resample(X, y)
    X_train = np.array(X)
    y_train = np.array(y)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    return X_train, y_train, lens


def test(estimator, filepath, trainingfile):
    """
    Test all the project vector files in the test directory separately and write them into the corresponding result files
    :param estimator: The best model trained
    :param filepath: test set file path
    :param trainingfile: The training data set used to train the model, here is only used to name the output result file
    :return: None
    """
    dff = pd.read_csv(filepath, header=None)
    X1 = []  # 向量
    y1 = []  # 标签
    with open(testset) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            a = row[0:-1]
            b = []
            for x in a:
                b.append(float(x))
            X1.append(b)
            if row[-1].upper() == "TRUE":
                y1.append(1)
            elif row[-1].upper() == "FALSE":
                y1.append(0)
    X1, y1 = SMOTE(random_state=42).fit_resample(X1, y1)
    X_test = np.array(X1)
    y_test = np.array(y1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    output = open("REMS_Test_" + trainingfile.split("\\")[-1].split(".")[0] + ".txt", "a+")
    # 测试数据
    testing_project = filepath.split("\\")[-1].split(".")[0]
    print("Start testing : " + testing_project + "\n")
    float_arr = X_test.astype(np.float)
    raw_scores = estimator.predict(float_arr)
    pred = np.where(raw_scores > label_threshold, 1, 0)
    pred = pred.reshape(pred.shape[0])
    precision = precision_score(y_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    recall = recall_score(y_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    f1 = f1_score(y_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    output.write(
        "-----------------------------------------------------------------------------------------------\n")
    output.write("Testing data : " + testing_project + "\n")
    output.write("Testing time : " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    output.write("precision: {:.3}, recall: {:.3}, f1:{:.3}".format(precision, recall, f1) + "\n")
    output.write(
        "-----------------------------------------------------------------------------------------------\n\n\n")
    output.close()


def train(X_train, y_train, lens):
    """
    Use grid search to find optimal parameters and then test the optimal model
    :param X_train:
    :param y_train:
    :param lens:
    :return: None
    """
    epoch = 30
    batch_size = 64
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv1D(16, 4, activation="relu", input_shape=(lens, 1)),
            tf.keras.layers.MaxPooling1D(3, 3),
            tf.keras.layers.Conv1D(32, 4, activation="relu"),
            tf.keras.layers.MaxPooling1D(3, 3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # model fitting
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size)
    # save model
    print("Saving model to disk \n")
    model_path = _training_data_path.split("\\")[-1].split(".")[0] + "_model"
    model.save(model_path)
    load_model = tf.keras.models.load_model(model_name)
    test(load_model, _REMS_project_path, _training_data_path)


if __name__ == '__main__':
    X_train, y_train, lens = load_data(_training_data_path)
    train(X_train, y_train, lens)
