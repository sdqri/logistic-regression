import math

import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressionClassifier:
    def __init__(self, alpha=0.0001):
        self.alpha = 0.000001
        self.cost_list = None

    def fit(self, x, y, threshold=0.1):
        #adding one column with value of 1 for B0
        x = np.hstack((np.matrix(np.ones(x.shape[0])).T, x))
        self.beta = np.zeros((x.shape[1], 1))
        self.x = x
        self.y = y
        self.beta, self.cost_list = self.gradient_descent(x, y, threshold)

    def predict(self, x):
        h_value = self.h(x, self.beta).item()
        return self.map_h(h_value)
    def h(self, x, beta):
        return 1 / (1 + np.exp(-x.dot(beta)))

    def calc_decent(self, x, y, beta):
        return ((self.h(x, beta) - y.reshape(x.shape[0], -1)).T.dot(x)).T

    def step_gradient(self, x, y, beta):
        beta = beta -  self.alpha * self.calc_decent(x, y, beta)
        return beta

    def cost(self, x, y, beta):
        p1 = np.dot(y.T, np.log(self.h(x, beta) + 0.000001))
        p2 = np.dot((1 - y).T, np.log(1 - self.h(x, beta) + 0.000001))
        j = -p1-p2
        return j.mean()

    def gradient_descent(self, x, y, threshold):
        beta = self.beta
        cd = threshold + 1
        old_cost = math.inf
        new_cost = 0
        temp = []
        while(cd > threshold):
            beta = self.step_gradient(x, y, beta)
            old_cost = new_cost
            new_cost = self.cost(x, y, beta)
            cd = abs(old_cost - new_cost)
            temp.append(new_cost)
        return (beta, temp)

    def plot_cost_list(self, ax):
        sns.lineplot(range(0, len(self.cost_list)), self.cost_list, ax=ax)

    def map_h(self, h_value):
        if(h_value<0.5):
            return 0
        else:
            return 1

    def score(self, x, y):
        #adding one column with value of 1 for B0
        x = np.hstack((np.matrix(np.ones(x.shape[0])).T, x))
        err = 0
        y = y.reshape(x.shape[0], -1)
        for i in range(x.shape[0]):
            if(self.predict(x[i,:]) != y[i]):
                err += 1
        err = err/x.shape[0]
        return err
