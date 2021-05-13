import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import gradient

class Data(object):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.data_str = open(path, mode='r').read()
        self.data_vec = list(self.data_str)
        self.chars = list(set(self.data_str))
        self.char_map = dict((c,i) for i,c in enumerate(self.chars))
        self.int2char_map = dict((i, c) for i,c in enumerate(self.chars))

class RNN(object):
    def __init__(self, data: Data, **kwargs) -> None:
        super().__init__()
        self.data = data
        self.m = kwargs['m'] if 'm' in kwargs else 100
        self.eta = kwargs['eta'] if 'eta' in kwargs else .1
        self.seq_len = kwargs['seq_len'] if 'seq_len' in kwargs else 25
        self.K = len(self.data.chars)
        self.init_params(kwargs['sig'] if 'sig' in kwargs else .01)

    def init_params(self, sig):
        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))
        self.U = np.random.normal(0, sig, size=(self.m, self.K))
        self.W = np.random.normal(0, sig, size=(self.m, self.m))
        self.V = np.random.normal(0, sig, size=(self.K, self.m))

        self.ada = [np.zeros_like(val) for val in (self.b, self.c, self.U, self.W, self.V)]

    def evaluate(self, h, x):
        a = np.dot(self.W, h) + np.dot(self.U, x) + self.b
        h = np.tanh(a)
        o = np.dot(self.V, h) + self.c
        p = np.exp(o) / np.sum(np.exp(o), axis=0) # softmax
        return a, h, o, p

    def synthetize(self, h0, x0, n):
        next_chars = np.zeros((len(self.data.chars), 1))
        next_chars[x0] = 1
        txt = ''
        for t in range(n):
            _, _, _, p = self.evaluate(h0, next_chars)
            input = np.random.choice(range(len(self.data.chars)), p=p.flat)
            next_chars = np.zeros((len(self.data.chars), 1))
            next_chars[input] = 1
            txt += self.data.int2char_map[input]
        return txt

    def compute_grads(self, inputs, targets, hidden_prev):
        x_, a_, h_, o_, p_ = {}, {}, {}, {}, {}
        n = len(inputs)
        loss = 0
        h_[-1] = np.copy(hidden_prev)

        # Forward pass
        for t in range(n):
            x_[t] = np.zeros((len(self.data.chars), 1))
            x_[t][inputs[t]] = 1

            a_[t], h_[t], o_[t], p_[t] = self.evaluate(h_[t - 1], x_[t])

            loss += -np.log(p_[t][targets[t]][0])

        g_b = np.zeros_like(self.b)
        g_c = np.zeros_like(self.c)
        g_U = np.zeros_like(self.U)
        g_W = np.zeros_like(self.W)
        g_V = np.zeros_like(self.V)

        o = np.zeros_like(p_[0])
        h = np.zeros_like(h_[0])
        h_next = np.zeros_like(h_[0])
        a = np.zeros_like(a_[0])

        # Backward pass
        for t in range(n-1, -1, -1):
            o = np.copy(p_[t])
            o[targets[t]] -= 1

            g_V += np.dot(o, h_[t].T)
            g_c += o

            h = np.dot(self.V.T, o) + h_next
            a = np.multiply(h, (1 - h_[t] ** 2))

            g_U += np.dot(a, x_[t].T)
            g_W += np.dot(a, h_[t - 1].T)
            g_b += a

            h_next = np.dot(self.W.T,  a)

        g_b = np.clip(g_b, -5, 5) 
        g_c = np.clip(g_c, -5, 5) 
        g_U = np.clip(g_U, -5, 5) 
        g_W = np.clip(g_W, -5, 5) 
        g_V = np.clip(g_V, -5, 5) 

        h = h_[n - 1]

        return (g_b, g_c, g_U, g_W, g_V), loss, h

    def train(self, num_epochs=20):
        cursor = 0
        n = 0
        losses = []
        ep = 0
        while ep < num_epochs:
            if n == 0 or cursor >= (len(self.data.data_str) - rnn.seq_len - 1):
                if ep != 0: print(f'\033[93m##### Epoch {ep} completed ####\033[0m')
                hidden_prev = np.zeros((self.m, 1))
                cursor = 0
                ep += 1

            inputs = [data.char_map[char] for char in data.data_str[cursor:cursor + rnn.seq_len]]
            targets = [data.char_map[char] for char in data.data_str[cursor + 1:cursor + rnn.seq_len + 1]]

            gradients, loss, hidden_prev = rnn.compute_grads(inputs, targets, hidden_prev)

            if ep == 1 and n == 0:
                smooth_loss = loss
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss
            losses.append(smooth_loss)

                        # if n == 0:
                        #     rnn.CheckGrads(inputs, targets, hidden_prev)

            if n % 10000 == 0:
                gen = rnn.synthetize(hidden_prev, inputs[0], 200)
                print(f"\033[91m-- synthetized text at {n} its:\033[0m")
                print(gen)
                print(f'\033[91mSmooth loss: {smooth_loss}\033[0m')

            # AdaGrad
            for i in range(len(gradients)):
                self.ada[i] += np.power(gradients[i], 2)
            g_b, g_c, g_U, g_W, g_V = gradients
            self.b -= self.eta / np.sqrt(self.ada[0] + 1e-50) * g_b
            self.c -= self.eta / np.sqrt(self.ada[1] + 1e-50) * g_c
            self.U -= self.eta / np.sqrt(self.ada[2] + 1e-50) * g_U
            self.W -= self.eta / np.sqrt(self.ada[3] + 1e-50) * g_W
            self.V -= self.eta / np.sqrt(self.ada[4] + 1e-50) * g_V

            n += 1
            cursor += rnn.seq_len
        synthesized_text = self.synthetize(hidden_prev, inputs[0], 1000)
        print("-------------- generated text using best model ---------------")
        print(synthesized_text)
        print("--------------------------------------------------------------")

        fig, ax = plt.subplots()
        ax.plot(losses, 'o-', label="Training Data")
        ax.legend()
        ax.set(xlabel='Iterations', ylabel='Loss')
        ax.grid()
        plt.savefig("loss_plot.png")
        plt.show()



if __name__ == '__main__':
    data = Data('./goblet_book.txt')
    rnn = RNN(data)
    # hidden_prev = np.zeros((rnn.m, 1))
    # inputs = [data.char_map[char] for char in data.data_str[0:0 + rnn.seq_len]]
    # print(rnn.synthetize(hidden_prev, inputs, 200))
    rnn.train()