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
        W = [np.random.normal(self.mu, self.sigma, (self.hidden_nodes[0], self.d))]
        for i in range(1, len(self.hidden_nodes)):
            W.append(np.random.normal(self.mu, self.sigma, (self.hidden_nodes[i], self.hidden_nodes[i - 1])))
        W.append(np.random.normal(self.mu, self.sigma, (self.K, self.hidden_nodes[-1])))

        b = [np.zeros((size, 1)) for size in self.hidden_nodes]
        b.append(np.zeros((self.K, 1)))

        self.W, self.b = W, b

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def EvaluateClassifier(self, X):
        h = [X]
        s = []
        for layer in range(len(self.W)):
            s_idx = np.dot(self.W[layer], h[-1]) + self.b[layer]
            s.append(s_idx)
            h.append(np.maximum(0, s_idx))
        h.pop(0)

        return s, h, self.softmax(s[-1])

    def ComputeCost(self, X, Y, _lambda):
        _, _, p = self.EvaluateClassifier(X)

        regularization_term = _lambda * (np.sum(np.power(self.W1, 2)) + np.sum(np.power(self.W2, 2)))
        regularization_term = np.sum([_lambda * np.sum(np.power(self.W[l], 2)) for l in range(len(self.W))])
        cross_entropy_loss = - np.log(np.sum(np.prod((np.array(Y), p), axis=0), axis=0))

        N = np.shape(X)[1]

        return (1 / N) * np.sum(cross_entropy_loss) + regularization_term # J

    def ComputeAccuracy(self, X, y):
        _, _, p = self.EvaluateClassifier(X)
        pred_indexes = np.argmax(p, axis=0)
        return np.array(np.where(pred_indexes == np.array(y))).shape[1] / np.size(y)

    def ComputeGradients(self, x, y, lamb): # lamb goes baaaaa
        gradient_W = [np.zeros(np.shape(self.W[i])) for i in range(len(self.W))]
        gradient_b = [np.zeros(np.shape(self.b[i])) for i in range(len(self.W))]

        _, H, P = self.EvaluateClassifier(x, self.W, self.b)

        G_batch = - (y - P)
        for layer in range(self.hidden_nodes, 0, -1):
            gradient_W[layer] = np.dot(np.multiply((1 / np.shape(x)[1]), G_batch), np.transpose(H[layer - 1])) \
                                + 2 * np.multiply(lamb, self.W[layer])
            gradient_b[layer] = np.reshape(np.dot(np.multiply((1 / np.shape(x)[1]), G_batch), np.ones(np.shape(x)[1])),
                                       (gradient_b[layer].shape[0], 1))

            G_batch = np.dot(np.transpose(self.W[layer]), G_batch)
            H[layer - 1][H[layer - 1] <= 0] = 0
            G_batch = np.multiply(G_batch, H[layer - 1] > 0)

        gradient_W[0] = np.dot(np.multiply((1 / np.shape(x)[1]), G_batch), np.transpose(X)) + np.multiply(lamb, self.W[0])
        gradient_b[0] = np.reshape(np.dot(np.multiply((1 / np.shape(x)[1]), G_batch), np.ones(np.shape(x)[1])), self.b[0].shape)

        return gradient_W, gradient_b

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

    print('figure3')
    nnt = NeuralNet((train_X, train_Y, train_labels), mu, sigma, hidden_nodes=[10,10])

if __name__ == "__main__":
    main()
