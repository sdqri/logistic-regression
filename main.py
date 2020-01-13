import math

import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from lrm import LogisticRegressionClassifier

def cross_validate(x, y, k=5):
    # Stacking x and y horiontally
    m = np.hstack((x, y.reshape(x.shape[0],-1)))
    # Shuffling data to randomize their order
    np.random.shuffle(m)
    # Splitting x and y
    x = m[:, :-1]
    y = m[:, -1].reshape(x.shape[0],-1)
    dl = len(y)
    fl = int(dl/k)
    folds_indices = [(i*fl, (i+1)*fl) for i in range(0, k)]
    scores = []
    for i in range(0, k):
        i, j = folds_indices[i]
        test_x = x[i:j, :]
        test_y = y[i:j, :]
        train_x = np.vstack((x[0:i, :], x[j:, :]))
        train_y = np.vstack((y[0:i, :], y[j:, :]))
        lrc_model = LogisticRegressionClassifier()
        lrc_model.fit(train_x, train_y)
        s = lrc_model.score(test_x, test_y)
        scores.append(s)
    return sum(scores) / len(scores)

if(__name__=="__main__"):
    #main
    names = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
 'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
 'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
 'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',
 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', "spam"]

    df = pd.read_csv("./spambase.data", header=None, names=names)
    df_features = df.iloc[:, 0:57]
    df_targets = df.loc[: , "spam"]
    x = df.iloc[:, 0:56].values
    y = df.loc[: , "spam"].values
    # Standardizing the features onto unit scale (mean = 0 and variance = 1)
    x_std = StandardScaler().fit_transform(x)
    # Adding 0.1 and logging all elements
    x_log = np.log(x + 0.1)
    # Binarizing data
    x_bin = np.where(x>0, 1, 0)

    lrc_normal = LogisticRegressionClassifier()
    lrc_normal.fit(x, y)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.set(xlabel="Iteration", ylabel="Cost", title="Unchanged Features")
    lrc_normal.plot_cost_list(ax)
    fig.savefig("lrc_normal.png")
    plt.close(fig)
    print("Unchanged Features - Misclassification Rate", cross_validate(x, y, k=5))
    print("Unchanged Features - iterations", len(lrc_normal.cost_list))

    lrc_std = LogisticRegressionClassifier()
    lrc_std.fit(x_std, y)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.set(xlabel="Iteration", ylabel="Cost", title="Standardized Features")
    lrc_std.plot_cost_list(ax)
    fig.savefig("lrc_std.png")
    plt.close(fig)
    print("Standardized Features - Misclassification Rate:", cross_validate(x_std, y, k=5))
    print("Standardized Features - iterations:", len(lrc_std.cost_list))

    lrc_log = LogisticRegressionClassifier()
    lrc_log.fit(x_log, y)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.set(xlabel="Iteration", ylabel="Cost", title="Logarithmize Data")
    lrc_log.plot_cost_list(ax)
    fig.savefig("lrc_log.png")
    plt.close(fig)
    print("Logarithmize Features - Misclassification Rate:", cross_validate(x_log, y, k=5))
    print("Logarithmize Features - iterations:", len(lrc_log.cost_list))

    lrc_bin = LogisticRegressionClassifier()
    lrc_bin.fit(x_bin, y)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.set(xlabel="Iteration", ylabel="Cost", title="Binarized Features")
    lrc_bin.plot_cost_list(ax)
    fig.savefig("lrc_bin.png")
    plt.close(fig)
    print("Binarized Features - Misclassification Rate:", cross_validate(x_bin, y, k=5))
    print("Binarized Features - iterations:", len(lrc_log.cost_list))
