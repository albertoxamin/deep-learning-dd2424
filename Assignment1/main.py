#!/usr/bin/python3
from os import X_OK, times
import pickle
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import std

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


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def EvaluateClassifier(X, W, b):
    s = np.add(np.matmul(W, X), b)
    return softmax(s)


def ComputeCost(X, Y, W, b, _lambda):
    p = EvaluateClassifier(X, W, b)
    return 1/np.size(X) * (-np.sum(Y*np.log(p))) + _lambda * np.sum(np.square(W))


def ComputeAccuracy(X, y, W, b):
    p = EvaluateClassifier(X, W, b)
    pred_indexes = np.argmax(p, axis=0)
    return np.array(np.where(pred_indexes == np.array(y))).shape[1] / np.size(y)


def ComputeGradsNumSlow(x, y, w, b, lamda, h):
    grad_w = np.zeros(w.shape)
    grad_b = np.zeros(b.shape)

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = ComputeCost(x, y, w, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(x, y, w, b_try, lamda)

        grad_b[i] = (c2-c1) / (2*h)

    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w_try = np.array(w)
            w_try[i, j] -= h
            c1 = ComputeCost(x, y, w_try, b, lamda)

            w_try = np.array(w)
            w_try[i, j] += h

            c2 = ComputeCost(x, y, w_try, b, lamda)

            grad_w[i, j] = (c2-c1) / (2*h)

    return grad_w, grad_b


def ComputeGradients(x, y, b, w, lamb, n_batch):
    p = EvaluateClassifier(x, w, b)
    g = -np.add(y,-p)
    grad_w = (1/n_batch) * np.matmul(g, np.array(x).T) + (2 * lamb * w)
    grad_b = np.array((1/n_batch)*np.matmul(g, np.ones(n_batch))
                      ).reshape(np.size(w, 0), 1)
    return grad_w, grad_b


def MiniBatchGD(X, Y, lab, val, test, W, b, n_batch=100, eta=0.001, epochs=100, _lambda=0.1):
    train_cost, train_acc, val_cost, val_acc = [], [], [], []
    batches_x, batches_y = [], []
    for i in range(X.shape[1]//n_batch):
        n = i*n_batch
        batches_x.append(X[:, n:n+n_batch])
        batches_y.append(Y[:, n:n+n_batch])
    for e in range(epochs):
        for ba in range(len(batches_x)):
            gw, gb = ComputeGradients(batches_x[ba], batches_y[ba], b, W, _lambda, n_batch)
            W -= (eta * gw)
            b -= (eta * gb)

        train_cost.append(ComputeCost(X, Y, W, b, _lambda))
        train_acc.append(ComputeAccuracy(X, lab, W, b))
        val_cost.append(ComputeCost(val[0], val[1], W, b, _lambda))
        val_acc.append(ComputeAccuracy(val[0], val[2], W, b))
        print(f"Epoch: {e}\t cost: {train_cost[-1]}\t train_acc: {train_acc[-1]}\t val_acc: {val_acc[-1]}")

    test_acc = ComputeAccuracy(test[0], test[2], W, b)
    epochs_label = np.arange(1, epochs+1, 1)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(epochs_label, train_cost, 'o-', label="Training Data")
    ax[0].plot(epochs_label, val_cost, label="Validation Data")
    ax[0].legend()
    ax[0].set(xlabel='Epochs', ylabel='Loss')
    ax[0].grid()
    ax[1].plot(epochs_label, train_acc, label="Training Data")
    ax[1].plot(epochs_label, val_acc, label="Validation Data")
    ax[1].plot(epochs_label[-1], test_acc, 'x-', label="Final Test Accuracy")
    ax[1].legend()
    ax[1].set(xlabel='Epochs', ylabel='Accuracy')
    ax[1].grid()
    fig.tight_layout() 
    plt.savefig(f'Result Pics/lambda_{_lambda}_epo_{epochs}_nbatch_{n_batch}_eta_{eta}.jpg')

    # plt.show()

    LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    w = np.array(W)
    # import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            im = w[i*5+j, :].reshape(32, 32, 3, order='F')
            sim = (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title(LABELS[i*5+j])
            ax[i][j].axis('off')
    plt.savefig(f'Result Pics/W_lambda_{_lambda}_epo_{epochs}_nbatch_{n_batch}_eta_{eta}.jpg')
    # plt.show()


def main():
    print("CIFAR-10-CLASSIFIER")
    train_X, train_Y, train_labels = LoadBatch(
        "../Datasets/cifar-10-batches-py/data_batch_1")
    val_X, val_Y, val_labels = LoadBatch(
        "../Datasets/cifar-10-batches-py/data_batch_2")
    test_X, test_Y, test_labels = LoadBatch(
        "../Datasets/cifar-10-batches-py/test_batch")
    mean_X, std_X = np.array([np.mean(train_X, 1)]).T, np.array(
        [np.std(train_X, 1)]).T

    train_X = norm(train_X, mean_X, std_X)
    val_X = norm(val_X, mean_X, std_X)
    test_X = norm(test_X, mean_X, std_X)
    mu, sigma = 0, 0.01
    W, b = np.random.normal(mu, sigma, (10, 3072)), np.random.normal(mu, sigma, (10, 1))
    MiniBatchGD(train_X, train_Y, train_labels, (val_X, val_Y, val_labels), (test_X, test_Y, test_labels), W, b, _lambda=0, epochs=40, n_batch=100, eta=0.1)
    W, b = np.random.normal(mu, sigma, (10, 3072)), np.random.normal(mu, sigma, (10, 1))
    MiniBatchGD(train_X, train_Y, train_labels, (val_X, val_Y, val_labels), (test_X, test_Y, test_labels), W, b, _lambda=0, epochs=40, n_batch=100, eta=0.001)
    W, b = np.random.normal(mu, sigma, (10, 3072)), np.random.normal(mu, sigma, (10, 1))
    MiniBatchGD(train_X, train_Y, train_labels, (val_X, val_Y, val_labels), (test_X, test_Y, test_labels), W, b, _lambda=.1, epochs=40, n_batch=100, eta=0.001)
    W, b = np.random.normal(mu, sigma, (10, 3072)), np.random.normal(mu, sigma, (10, 1))
    MiniBatchGD(train_X, train_Y, train_labels, (val_X, val_Y, val_labels), (test_X, test_Y, test_labels), W, b, _lambda=1, epochs=40, n_batch=100, eta=0.001)

if __name__ == "__main__":
    main()
