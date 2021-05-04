#!/usr/bin/python3
from os import X_OK, times
import pickle
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import std
import random

class NeuralNet:

    def __init__(self, data, mu, sigma, hidden_nodes, init_mode='default') -> None:
        self.data = data
        self.d = data[0].shape[0] # the dimension of the data (in this case the image size 32x32x3)
        self.K = data[1].shape[0] # The number of labels (should be 10)
        self.mu, self.sigma = mu, sigma # mean, variance
        self.hidden_nodes = hidden_nodes
        self.InitializeWeights(init_mode)

    def InitializeWeights(self, init_mode):
        if init_mode == 'he':
            W = [np.random.normal(0, (2 / self.hidden_nodes[0]), (self.hidden_nodes[0], self.d))]
            for i in range(1, len(self.hidden_nodes)):
                W.append(np.random.normal(0, (2 / self.hidden_nodes[i - 1]), (self.hidden_nodes[i], self.hidden_nodes[i - 1])))
            W.append(np.random.normal(0, (2 / self.hidden_nodes[-1]), (self.K, self.hidden_nodes[-1])))
        else:
            W = [np.random.normal(self.mu, self.sigma, (self.hidden_nodes[0], self.d))]
            for i in range(1, len(self.hidden_nodes)):
                W.append(np.random.normal(self.mu, self.sigma, (self.hidden_nodes[i], self.hidden_nodes[i - 1])))
            W.append(np.random.normal(self.mu, self.sigma, (self.K, self.hidden_nodes[-1])))

        b = [np.zeros((size, 1)) for size in self.hidden_nodes]
        b.append(np.zeros((self.K, 1)))

        self.W, self.b = W, b

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def EvaluateClassifier(self, X, batch_norm=False, training=False):
        if batch_norm:
            scores = []  # nonnormalized scores
            batch_norm_scores = []
            batch_norm_relu_scores = [X]
            mus = []  # means
            vars = []  # variabilitiess
            score_layer = None

            for l in range(len(self.W)):
                score_layer = np.dot(self.W[l], batch_norm_relu_scores[-1]) + self.b[l]
                scores.append(score_layer)

                # mean and variance calculations for each layer
                mus.append(np.mean(score_layer, axis=1))
                vars.append(np.var(score_layer, axis=1))

                # batch normalization calculations
                batch_norm = np.zeros(np.shape(score_layer))
                a = np.diag(np.power((vars[-1] + 1e-16), (-1 / 2))) # 1e-16 is to prevent zero division
                for i in range(np.shape(score_layer)[1]):
                    batch_norm[:, i] = np.dot(a, (score_layer[:, i] - mus[-1]))

                batch_norm_scores.append(batch_norm)
                batch_norm_relu_scores.append(np.maximum(0, batch_norm))
            batch_norm_relu_scores.pop(0)

            return self.softmax(score_layer), scores, batch_norm_scores, batch_norm_relu_scores, mus, vars
        else:
            h = [X]
            s = []
            for l in range(len(self.W)):
                s_idx = np.dot(self.W[l], h[-1]) + self.b[l]
                s.append(s_idx)
                h.append(np.maximum(0, s_idx))
            h.pop(0)

            return s, h, self.softmax(s[-1])

    def ComputeCost(self, X, Y, _lambda):
        _, _, p = self.EvaluateClassifier(X)

        regularization_term = np.sum([_lambda * np.sum(np.power(self.W[l], 2)) for l in range(len(self.W))])
        cross_entropy_loss = - np.log(np.sum(np.prod((np.array(Y), p), axis=0), axis=0))

        N = np.shape(X)[1]

        return (1 / N) * np.sum(cross_entropy_loss) + regularization_term # J

    def ComputeAccuracy(self, X, y):
        _, _, p = self.EvaluateClassifier(X)
        pred_indexes = np.argmax(p, axis=0)
        return np.array(np.where(pred_indexes == np.array(y))).shape[1] / np.size(y)

    def ComputeGradients(self, x, y, lamb, batch_norm=False): # lamb goes baaaaa
        gradient_W = [np.zeros(np.shape(w)) for w in self.W]
        gradient_b = [np.zeros(np.shape(b)) for b in self.b]
        if batch_norm:
            P, scores, bn_scores, bn_relu_scores, mus, vars = self.EvaluateClassifier(x, batch_norm=batch_norm)
            G_batch = -(y - P)
            l = len(self.W) - 1

            gradient_b.insert(0, (1 / np.shape(y)[1]) * np.sum(G_batch, axis=1).reshape(-1, 1))
            gradient_W.insert(0, (1 / np.shape(y)[1]) * np.dot(G_batch, bn_relu_scores[l-1].T) + 2 * lamb * self.W[l])

            G_batch = np.dot(np.transpose(self.W[l]), G_batch)
            G_batch = np.multiply(G_batch, list(map(lambda num: num > 0, bn_scores[l - 1])))

            for l in range(len(self.W) - 2, -1, -1):
                n = np.shape(G_batch)[1]
                var_layer = vars[l]
                mu_layer = mus[l]
                score_layer = scores[l]
                v1_2 = np.diag(np.power((var_layer + 1e-16), (-1 / 2)))
                v3_2 = np.diag(np.power((var_layer + 1e-16), (-3 / 2)))
                grad_var = -(1 / 2) * sum([np.dot(G_batch[:, i], (np.dot(v3_2, np.diag(score_layer[:, i] - mu_layer))))
                                        for i in range(n)])
                grad_mu = (-1) * sum([np.dot(G_batch[:, i], v1_2) for i in range(n)])
                G_temp = np.zeros(np.shape(G_batch))
                for i in range(n):
                    G_temp[:, i] = np.dot(G_batch[:, i], v1_2) + (2 / n) * \
                                np.dot(grad_var, np.diag(score_layer[:, i] - mu_layer)) + (1 / n) * grad_mu
                if l > 0:
                    prev_layer = bn_relu_scores[l - 1]
                else:
                    prev_layer = x
                gradient_b.insert(0, (1 / n) * np.sum(G_temp, axis=1).reshape(-1, 1))
                gradient_W.insert(0, (1 / n) * np.dot(G_temp, prev_layer.T) + 2 * lamb * self.W[l])

                if l > 0:
                    G_batch = np.dot(np.transpose(self.W[l]), G_batch)
                    G_batch = np.multiply(G_batch, list(map(lambda num: num > 0, bn_scores[l - 1])))

            return gradient_W, gradient_b, mus, vars
        else:
            _, H, P = self.EvaluateClassifier(x)
            G_batch = (P - y)
            for l in range(len(self.hidden_nodes), 0, -1):
                gradient_W[l] = (1/x.shape[1])*np.matmul(G_batch, H[l-1].T) + 2 * lamb * self.W[l] # Equation (22)
                gradient_b[l] = (1/x.shape[1])*np.matmul(G_batch, np.diag(np.eye(x.shape[1]))) # Equation (22)
                gradient_b[l] = np.reshape(gradient_b[l], self.b[l].shape) # replace with -1?

                G_batch = np.matmul(self.W[l].T, G_batch) # Equation (23)
                H[l-1][H[l-1] <= 0] = 0 # ReLU
                G_batch = np.multiply(G_batch, H[l-1] > 0) # Equation (24)

            gradient_W[0] = (1/x.shape[1])*np.matmul(G_batch, x.T) + 2 * lamb * self.W[0]
            gradient_b[0] = np.reshape((1/x.shape[1])*np.matmul(G_batch, np.diag(np.eye(x.shape[1]))), self.b[0].shape)

            return gradient_W, gradient_b

    def ComputeGradsNum(self, x, y, lamb, h):
        gradient_W = [np.zeros(np.shape(w)) for w in self.W]
        gradient_b = [np.zeros(np.shape(b)) for b in self.b]

        cost = self.ComputeCost(x, y, lamb)
        import copy
        for k in range(len(self.W)):
            for i in range(len(self.b[k])):
                temp = copy.deepcopy(self.b)
                temp[k][i] += h
                cost_2 = self.ComputeCost(x, y, lamb)
                gradient_b[k][i] = (cost_2 - cost) / h

            for i in range(self.W[k].shape[0]):
                for j in range(self.W[k].shape[1]):
                    temp = copy.deepcopy(self.W)
                    temp[k][i, j] += h
                    cost_2 = self.ComputeCost(x, y, lamb)
                    gradient_W[k][i, j] = (cost_2 - cost) / h

        return gradient_W, gradient_b


    def SanityCheck(self):
        x, y = self.data[0][:, 0:1], self.data[1][:, 0:1]
        anw, anb  = self.ComputeGradients(x, y, 0)
        print("an-done")
        numw, numb = self.ComputeGradsNum(x, y, 0, 1e-5)
        print("num-done")
        
        for i in range(len(anw)):
            print(f"W diff:{abs(np.mean(anw[i]) - np.mean(numw[i]))}\t b diff:{abs(np.mean(anb[i]) - np.mean(numb[i]))}")

    def MiniBatchGD(self, val, test, n_batch=100, eta={}, epochs=100, _lambda=0.1, batch_norm=False, verbose=True):
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
        self.moving_avg_mean = [np.zeros(l) for l in self.hidden_nodes]
        self.moving_avg_var = [np.zeros(l) for l in self.hidden_nodes]
        alpha = 0
        # print (len(batches_x))
        for e in range(epochs):
            for ba in range(len(batches_x)):
                if batch_norm:
                    gw, gb, gm, gv = self.ComputeGradients(batches_x[ba], batches_y[ba], _lambda, batch_norm=batch_norm)
                    for layer in range(len(self.hidden_nodes)):
                        self.moving_avg_mean[layer] = alpha * self.moving_avg_mean[layer] + (1 - alpha) * gm[layer]
                        self.moving_avg_var[layer] = alpha * self.moving_avg_var[layer] + (1 - alpha) * gv[layer]
                    alpha = 0.99
                else:
                    gw, gb = self.ComputeGradients(batches_x[ba], batches_y[ba], _lambda)
                comp_eta = compute_eta(t, eta['min'], eta['max'], eta['step_size'], eta['l'])
                for l in range(len(self.hidden_nodes)):
                    self.W[l] -= (comp_eta * gw[l])
                    self.b[l] -= (comp_eta * gb[l])
                t += 1
            c = list(zip(batches_x, batches_y))
            random.shuffle(c)
            batches_x, batches_y = zip(*c)
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
        print("Final Test Accuracy", test_acc)
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

    # nnt = NeuralNet((train_X[:, 0:100], train_Y[:, 0:100], train_labels[:, 0:100]), mu, sigma, hidden_nodes=[10])
    # nnt.SanityCheck()

    # nnt = NeuralNet((train_X, train_Y, train_labels), mu, sigma, hidden_nodes=[50,50], init_mode='he')
    # nnt.MiniBatchGD((val_X, val_Y, val_labels), (test_X, test_Y, test_labels), _lambda=0.008399842677560622, epochs=40, n_batch=100, eta={
    #     'min': 1e-5,
    #     'max': 1e-1,
    #     'step_size': 800, 'l': 0
    # })

    # nnt = NeuralNet((train_X, train_Y, train_labels), mu, sigma, hidden_nodes=[50,50], init_mode='he')
    # nnt.MiniBatchGD((val_X, val_Y, val_labels), (test_X, test_Y, test_labels), _lambda=0.005, epochs=60, n_batch=100, eta={
    #     'min': 1e-5,
    #     'max': 1e-1,
    #     'step_size': 2250, 'l': 0
    # })
    # poor performer
    # nnt = NeuralNet((train_X, train_Y, train_labels), mu, sigma, hidden_nodes=[50, 30, 20, 20, 10, 10, 10, 10], init_mode='he')
    # nnt.MiniBatchGD((val_X, val_Y, val_labels), (test_X, test_Y, test_labels), _lambda=0.005, epochs=60, n_batch=100, eta={
    #     'min': 1e-5,
    #     'max': 1e-1,
    #     'step_size': 2250, 'l': 0
    # })
    # nnt = NeuralNet((train_X, train_Y, train_labels), mu, sigma, hidden_nodes=[50,50], init_mode='he')
    # nnt.MiniBatchGD((val_X, val_Y, val_labels), (test_X, test_Y, test_labels), _lambda=0.005, epochs=60, n_batch=100, eta={
    #     'min': 1e-5,
    #     'max': 1e-1,
    #     'step_size': 2250, 'l': 0
    # }, batch_norm=True)
    # nnt = NeuralNet((train_X, train_Y, train_labels), mu, sigma, hidden_nodes=[50, 30, 20, 20, 10, 10, 10, 10], init_mode='he')
    # nnt.MiniBatchGD((val_X, val_Y, val_labels), (test_X, test_Y, test_labels), _lambda=0.0000075, epochs=60, n_batch=100, eta={
    #     'min': 1e-5,
    #     'max': 1e-1,
    #     'step_size': 2250, 'l': 0
    # }, batch_norm=True)

    sigmas = [1e-1, 1e-3, 1e-4]
    network = [50, 50]
    for sig in sigmas:
        print("BATCH NORM OFF")
        nnt = NeuralNet((train_X, train_Y, train_labels), mu, sigma=sig, hidden_nodes=network)
        nnt.MiniBatchGD((val_X, val_Y, val_labels), (test_X, test_Y, test_labels), _lambda=0.0000075, epochs=20, n_batch=100, eta={
            'min': 1e-5,
            'max': 1e-1,
            'step_size': 2250, 'l': 0
        }, batch_norm=False)
        print("BATCH NORM ON")
        nnt = NeuralNet((train_X, train_Y, train_labels), mu, sigma=sig, hidden_nodes=network)
        nnt.MiniBatchGD((val_X, val_Y, val_labels), (test_X, test_Y, test_labels), _lambda=0.0000075, epochs=20, n_batch=100, eta={
            'min': 1e-5,
            'max': 1e-1,
            'step_size': 2250, 'l': 0
        }, batch_norm=True)

if __name__ == "__main__":
    main()
