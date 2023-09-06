# -*- encoding = utf-8 -*-
"""
@description: 用RF训练并对REMS数据进行测试
@date: 2022/9/27
@File : RF.py
@Software : PyCharm
"""
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
import sys
import time
import datetime
import csv
import os
import pathlib

# _training_data_path = sys.argv[1]  # Training dataset file path
# _REMS_project_path = sys.argv[2]  # Test dataset file path
# _output_write_path = sys.argv[3]  # path of output folder


def load_data(src):
    """
    加载训练数据集并划分训练集和测试集
    :param src: 训练集文件
    :return: 训练集的vec和label
    """
    df = pd.read_csv(src, header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # 定义SMOTE模型，random_state相当于随机数种子的作用
    smo = SMOTE(random_state=42)
    X_train, y_train = smo.fit_resample(X, y)
    print("data loaded for src: " + src)
    return X_train, y_train


def test(estimator, filepath, outputPath):
    """
    分别测试测试目录下的所有项目向量文件并写入对应的结果文件中
    :param estimator: 训练得到的最优模型
    :param filepath: 测试集文件路径
    :param trainingfile: 训练模型所使用的训练数据集，这里仅用于对输出结果文件命名
    :return: None
    """
    dff = pd.read_csv(filepath, header=None)
    X_test = dff.iloc[:, :-1]
    y_test = dff.iloc[:, -1]
    X_test, y_test = SMOTE(random_state=42).fit_resample(X_test, y_test)
    # output = open("REMS_Test_" + trainingfile.split("\\")[-1].split(".")[0] + ".txt", "a+")
    output = open(outputPath, "a+")
    # Test Data
    testing_emb1 = filepath.split("\\")[-2]
    testing_project = filepath.split("\\")[-1].split(".")[0]
    testing_file = filepath.split("\\")[-3].split(".")[0]
    print("Start testing : " + testing_project + "\n")
    y_pre = estimator.predict(X_test)
    precision = precision_score(y_test, y_pre, labels=None, pos_label=1, average='binary', sample_weight=None)
    recall = recall_score(y_test, y_pre, labels=None, pos_label=1, average='binary', sample_weight=None)
    f1 = f1_score(y_test, y_pre, labels=None, pos_label=1, average='binary', sample_weight=None)
    output.write(
        "-----------------------------------------------------------------------------------------------\n")
    output.write("Testing emb1 : " + testing_emb1 + "\n")
    output.write("Testing emb2 : " + testing_project + "\n")
    output.write("Testing method : " + testing_file + "\n")
    output.write("Testing time : " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    output.write("precision: {:.3}, recall: {:.3}, f1:{:.3}".format(precision, recall, f1) + "\n")
    output.write(
        "-----------------------------------------------------------------------------------------------\n\n\n")
    output.close()
    return testing_emb1, testing_project, precision, recall, f1


def train(X_train, y_train):
    """
    运用网格搜索寻找最优参数，再对最优模型进行测试
    :param X_train:
    :param y_train:
    :return: None
    """
    # 网格搜索参数列表
    tuned_parameters = [
        {'n_estimators': [80, 90, 100, 50, 60, 70, 40, 20, 30, 65], 'max_features': [8, 12, 10, 15, 17, 5]},
        {'bootstrap': [False], 'n_estimators': [3, 7, 10], 'max_features': [2, 3, 4]},
    ]
    # 生成模型
    print("Start trainging : " + "\n")
    grid = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='roc_auc', verbose=2, n_jobs=4)
    # 把数据交给模型训练
    model = grid.fit(X_train, y_train)
    return model
    # test(model, _REMS_project_path, _training_data_path)


def runAllTests():
    training_csv = "C:\\Users\\devso\\OneDrive\\Desktop\\Coding\\DSCI-644\\Project\\project-dsci-644-group-9-group-9\\rems_modified\\Training_CSV"
    test_data = "C:\\Users\\devso\\OneDrive\\Desktop\\Coding\\DSCI-644\\Project\\project-dsci-644-group-9-group-9\\rems_modified\\GEMS_test_data"
    output_file = "C:\\Users\\devso\\OneDrive\\Desktop\\Coding\\DSCI-644\\Project\\project-dsci-644-group-9-group-9\\rems_modified\\outputs\\KNN_test.txt"
    count = 0
    testTrainDict = {}
    for embFolder in os.listdir(training_csv):
        embFolderDir = training_csv + '\\' + embFolder
        embeddings = pathlib.Path(embFolderDir)
        for emb_csv in embeddings.iterdir():
            testTrainDict[str(emb_csv)] = []

    for embFolder in os.listdir(training_csv):
        embFolderDir = training_csv + '\\' + embFolder
        embeddings = pathlib.Path(embFolderDir)
        for emb_csv in embeddings.iterdir():
            for testMethod in os.listdir(test_data):
                testMethodDir = test_data + '\\' + testMethod
                # print(testMethodDir)
                testPath = testMethodDir + '\\' + str(emb_csv).split('\\')[-2] + '\\' + str(emb_csv).split('\\')[-1]
                testTrainDict[str(emb_csv)].append(testPath)

    # print(json.dumps(testTrainDict, sort_keys=True, indent=4))

    outs = []
    for key in testTrainDict:
        if key.split("\\")[-1].split(".")[0] == 'prone_cg':
            continue
        X_train, y_train = load_data(key)
        model = train(X_train, y_train)
        print("src: " + key)
        print("test_method: " + testTrainDict[key][1])
        emb1, emb2, p, r, f1 = test(model, testTrainDict[key][0], output_file)
        outs.append(['RF', emb1, emb2, p, r, f1])
        # for test_method in testTrainDict[key]:
        #     print("src: " + key)
        #     print("test_method: " + test_method)
        #     test(model, test_method, output_file)
    write_to_csv(outs)


def write_to_csv(outs):
    output_csv = "C:\\Users\\devso\\OneDrive\\Desktop\\Coding\\DSCI-644\\Project\\project-dsci-644-group-9-group-9\\rems_modified\\outputs\\RF_results.csv"
    with open(output_csv, 'w', newline='') as out_csv:
        csv_writer = csv.writer(out_csv)
        for out in outs:
            csv_writer.writerow(out)


if __name__ == '__main__':
    start_time = time.time()
    runAllTests()
    # X_train, y_train = load_data(_training_data_path)
    # train(X_train, y_train)
    print("\n\n\nTESTING COMPLETED IN " + str(time.time() - start_time) + " seconds.")