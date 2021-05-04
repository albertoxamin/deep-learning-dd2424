#!/usr/bin/python3
from os import X_OK, times
import pickle
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import std
import random

class NeuralNet:

    def __init__(self, data, mu, sigma, hidden_nodes) -> None:
        self.data = data
        self.d = data[0].shape[0] # the dimension of the data (in this case the image size 32x32x3)
        self.K = data[1].shape[0] # The number of labels (should be 10)
        self.mu, self.sigma = mu, sigma # mean, variance
        self.hidden_nodes = hidden_nodes
        self.InitializeWeights()

    def InitializeWeights(self):
        self.W1, self.b1 = np.random.normal(self.mu, self.sigma, (self.hidden_nodes, self.d)), np.random.normal(self.mu, self.sigma, (self.hidden_nodes, 1))
        self.W2, self.b2 = np.random.normal(self.mu, self.sigma, (self.K, self.hidden_nodes)), np.random.normal(self.mu, self.sigma, (self.K, 1))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def EvaluateClassifier(self, X):
        s1 = np.add(np.matmul(self.W1, X), self.b1)
        h = np.maximum(0, s1)
        s = np.add(np.matmul(self.W2, h), self.b2)
        return s1, h, self.softmax(s)

    def ComputeCost(self, X, Y, _lambda):
        _, _, p = self.EvaluateClassifier(X)

        regularization_term = _lambda * (np.sum(np.power(self.W1, 2)) + np.sum(np.power(self.W2, 2)))
        cross_entropy_loss = - np.log(np.sum(np.prod((np.array(Y), p), axis=0), axis=0))

        N = np.shape(X)[1]
        return (1 / N) * np.sum(cross_entropy_loss) + regularization_term # J

    def ComputeAccuracy(self, X, y):
        _, _, p = self.EvaluateClassifier(X)
        pred_indexes = np.argmax(p, axis=0)
        return np.array(np.where(pred_indexes == np.array(y))).shape[1] / np.size(y)

    def ComputeGradients(self, x, y, lamb):
        s1, h, p = self.EvaluateClassifier(x)

        grad_W1 = np.zeros(np.shape(self.W1))
        grad_b1 = np.zeros(np.shape(self.b1))
        grad_W2 = np.zeros(np.shape(self.W2))
        grad_b2 = np.zeros(np.shape(self.b2))

        # Backward Pass
        for i in range(np.shape(x)[1]):
            Y_i = y[:, i].reshape((-1, 1))
            P_i = p[:, i].reshape((-1, 1))
            X_i = x[:, i].reshape((-1, 1))
            hidden_i = h[:, i].reshape((-1, 1))
            s_i = s1[:, i]

            temp_g = P_i - Y_i
            grad_b2 = grad_b2 + temp_g
            grad_W2 = grad_W2 + np.dot(temp_g, hidden_i.T)


            temp_g = np.dot(self.W2.T, temp_g)
            temp_g = np.dot(np.diag(list(map(lambda num: num > 0, s_i))), temp_g)


            grad_b1 = grad_b1 + temp_g
            grad_W1 = grad_W1 + np.dot(temp_g, X_i.T)

        grad_b1 = np.divide(grad_b1, np.shape(x)[1])
        grad_W1 = np.divide(grad_W1, np.shape(x)[1]) + 2 * lamb * self.W1

        grad_b2 = np.divide(grad_b2, np.shape(x)[1])
        grad_W2 = np.divide(grad_W2, np.shape(x)[1]) + 2 * lamb * self.W2

        return grad_W1, grad_b1, grad_W2, grad_b2

    def ComputeGradsNum(self, x, y, lamb, h):
        W = [self.W1, self.W2]
        b = [self.b1, self.b2]

        # initialize gradients
        grad_W = []
        grad_b = []

        for i in range(len(W)):
            grad_W.append(np.zeros(np.shape(W[i])))
            grad_b.append(np.zeros(np.shape(b[i])))

        cost = self.ComputeCost(x, y, lamb)
        import copy
        for k in range(len(W)):
            for i in range(len(b[k])):
                b_try = copy.deepcopy(b)
                b_try[k][i] += h
                self.b1, self.b2 = b_try[0], b_try[1]
                cost2 = self.ComputeCost(x, y, lamb)
                grad_b[k][i] = (cost2 - cost) / h
            self.b1, self.b2 = b[0], b[1]
            for i in range(W[k].shape[0]):
                for j in range(W[k].shape[1]):
                    W_try = copy.deepcopy(W)
                    W_try[k][i, j] += h
                    self.W1, self.W2 = W_try[0], W_try[1]
                    cost2 = self.ComputeCost(x, y, lamb)
                    grad_W[k][i, j] = (cost2 - cost) / h
            self.W1, self.W2 = W[0], W[1]

        return grad_W[0], grad_b[0], grad_W[1], grad_b[1]

    def SanityCheck(self):
        x, y = self.data[0][:, 0:1], self.data[1][:, 0:1]
        analytical = self.ComputeGradients(x, y, 0)
        numerical = self.ComputeGradsNum(x, y, 0, 0.1)
        
        for i in range(analytical):
            absdiff = abs(np.mean(analytical[i]) - np.mean(numerical[i]))
            print(absdiff)

    def MiniBatchGD(self, val, test, n_batch=100, eta={}, epochs=100, _lambda=0.1, verbose=True):
        def compute_eta(t, eta_min, eta_max, step_size, l):
            t = t % (2* step_size)
            if t <= step_size:
                return eta_min + ((t - 2*l*step_size)/step_size)*(eta_max - eta_min)
            elif t <= 2 * step_size:
                return eta_max - ((t - (2*l+1)*step_size)/step_size)*(eta_max - eta_min)

        train_cost, train_acc, val_cost, val_acc = [], [], [], []
        batches_x, batches_y = [], []
        for i in range(self.data[0].shape[1]//n_batch):
            n = i*n_batch
            batches_x.append(self.data[0][:, n:n+n_batch])
            batches_y.append(self.data[1][:, n:n+n_batch])

        t = 0
        # etas = []
        # for i in range(6*eta['step_size']):
        #     comp_eta = compute_eta(i, eta['min'], eta['max'], eta['step_size'])
        #     etas.append(comp_eta)
        #     # print(comp_eta)
        # plt.plot(etas)
        # plt.show()
        # return
        update_steps = []
        # print (len(batches_x))
        for e in range(epochs):
            for ba in range(len(batches_x)):
                gw1, gb1, gw2, gb2 = self.ComputeGradients(batches_x[ba], batches_y[ba], _lambda)
                comp_eta = compute_eta(t, eta['min'], eta['max'], eta['step_size'], eta['l'])
                self.W1 -= (comp_eta * gw1)
                self.b1 -= (comp_eta * gb1)
                self.W2 -= (comp_eta * gw2)
                self.b2 -= (comp_eta * gb2)
                t += 1

            if verbose:
                train_cost.append(self.ComputeCost(self.data[0], self.data[1], _lambda))
                train_acc.append(self.ComputeAccuracy(self.data[0], self.data[2]))
                val_cost.append(self.ComputeCost(val[0], val[1], _lambda))
                val_acc.append(self.ComputeAccuracy(val[0], val[2]))
                update_steps.append(t)
                print(f"Epoch: {e}\t cost: {train_cost[-1]}\t train_acc: {train_acc[-1]}\t val_acc: {val_acc[-1]}")

        test_acc = self.ComputeAccuracy(test[0], test[2])
        # return test_acc
        epochs_label = np.arange(1, epochs+1, 1)
        epochs_label = update_steps
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(epochs_label, train_cost, 'o-', label="Training Data")
        ax[0].plot(epochs_label, val_cost, label="Validation Data")
        ax[0].legend()
        ax[0].set(xlabel='Update Steps', ylabel='Loss')
        ax[0].grid()
        ax[1].plot(epochs_label, train_acc, label="Training Data")
        ax[1].plot(epochs_label, val_acc, label="Validation Data")
        ax[1].plot(epochs_label[-1], test_acc, 'x-', label="Final Test Accuracy")
        ax[1].legend()
        ax[1].set(xlabel='Update Steps', ylabel='Accuracy')
        ax[1].grid()
        fig.tight_layout()
        plt.show()
        # plt.savefig(f'Result Pics/lambda_{_lambda}_epo_{epochs}_nbatch_{n_batch}_eta_{eta}.jpg')

        # plt.show()



def LoadBatch(filename, K=10):
    dict = {}
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    x = np.array(dict.get(b'data'), dtype=float).T
    y_labels = np.array([dict.get(b'labels')])
    y_one_hot = np.zeros((K, x.shape[1]))

    y_one_hot[y_labels, np.arange(y_labels.size)] = 1

    return x, y_one_hot, y_labels


def norm(X, mean, std):
    return np.divide(np.subtract(X, mean), std)

def main():
    print("CIFAR-10-CLASSIFIER")
    train_X, train_Y, train_labels = LoadBatch("../Datasets/cifar-10-batches-py/data_batch_1")
    val_X, val_Y, val_labels = LoadBatch("../Datasets/cifar-10-batches-py/data_batch_2")
    # for i in range(4):
    #     X, Y, labels = LoadBatch(
    #     f"../Datasets/cifar-10-batches-py/data_batch_{i+2}")
    #     train_X = np.concatenate((train_X, X), axis=1)
    #     train_Y = np.concatenate((train_Y, Y), axis=1)
    #     train_labels = np.concatenate((train_labels, labels), axis=1)

    # val_size = 1000
    # val_X, val_Y, val_labels = train_X[:, -val_size:], train_Y[:, -val_size:], train_labels[:, -val_size:]
    # train_X, train_Y, train_labels = train_X[:, :-val_size], train_Y[:, :-val_size], train_labels[:, :-val_size]
    test_X, test_Y, test_labels = LoadBatch(
        "../Datasets/cifar-10-batches-py/test_batch")
    mean_X, std_X = np.array([np.mean(train_X, 1)]).T, np.array(
        [np.std(train_X, 1)]).T
    print(train_X.shape)
    print(val_X.shape)


    train_X = norm(train_X, mean_X, std_X)
    val_X = norm(val_X, mean_X, std_X)
    test_X = norm(test_X, mean_X, std_X)
    mu, sigma = 0, 0.01
    # W, b = np.random.normal(mu, sigma, (10, 3072)), np.random.normal(mu, sigma, (10, 1))

    # compare(train_X, train_Y, W, b, n_batch=100)

    # nnt = NeuralNet((train_X[:, 0:100], train_Y[:, 0:100], train_labels[:, 0:100]), mu, sigma, hidden_nodes=50)
    # nnt.SanityCheck()

    # print('figure3')
    # nnt = NeuralNet((train_X, train_Y, train_labels), mu, sigma, hidden_nodes=50)
    # nnt.MiniBatchGD((val_X, val_Y, val_labels), (test_X, test_Y, test_labels), _lambda=0.01, epochs=10, n_batch=100, eta={
    #     'min': 1e-5,
    #     'max': 1e-1,
    #     'step_size': 500, 'l': 0
    # }) # graph figure 3

    # print('figure4')
    # nnt = NeuralNet((train_X, train_Y, train_labels), mu, sigma, hidden_nodes=50)
    # nnt.MiniBatchGD((val_X, val_Y, val_labels), (test_X, test_Y, test_labels), _lambda=0.01, epochs=50, n_batch=100, eta={
    #     'min': 1e-5,
    #     'max': 1e-1,
    #     'step_size': 800, 'l': 0
    # }) # graph figure 4
    # return

    ## coarse search
    # l_min = -5
    # l_max = -1
    # ls = [l_min + (l_max - l_min)*random.random() for i in range(10)]
    # for l in ls:
    #     lambd = 10**l
    #     nnt = NeuralNet((train_X, train_Y, train_labels), mu, sigma, hidden_nodes=50)
    #     score = nnt.MiniBatchGD(
    #         (val_X, val_Y, val_labels),
    #         (test_X, test_Y, test_labels),
    #         _lambda=lambd, epochs=20, n_batch=100,
    #         eta={
    #             'min': 1e-5,
    #             'max': 1e-1,
    #             'step_size': 800,
    #             'l': 0
    #         }, verbose=False)
    #     print('l:', l, 'λ:', lambd, '\t score:', score)

    ## finer grained search
    # l_min = -2.1
    # l_max = -1.9
    # ls = [l_min + (l_max - l_min)*random.random() for i in range(10)]
    # for l in ls:
    #     lambd = 10**l
    #     nnt = NeuralNet((train_X, train_Y, train_labels), mu, sigma, hidden_nodes=50)
    #     score = nnt.MiniBatchGD(
    #         (val_X, val_Y, val_labels),
    #         (test_X, test_Y, test_labels),
    #         _lambda=lambd, epochs=20, n_batch=100,
    #         eta={
    #             'min': 1e-5,
    #             'max': 1e-1,
    #             'step_size': 800,
    #             'l': 0
    #         }, verbose=False)
    #     print('l:', l, 'λ:', lambd, '\t score:', score)

    nnt = NeuralNet((train_X, train_Y, train_labels), mu, sigma, hidden_nodes=50)
    nnt.MiniBatchGD((val_X, val_Y, val_labels), (test_X, test_Y, test_labels), _lambda=0.008399842677560622, epochs=40, n_batch=100, eta={
        'min': 1e-5,
        'max': 1e-1,
        'step_size': 800, 'l': 0
    }) # final network

if __name__ == "__main__":
    main()
