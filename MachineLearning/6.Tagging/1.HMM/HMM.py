# -*-coding:UTF-8-*-
'''
    Created on 2015-12-10
    @author: Liu Weijie
    reference:
        [1] <Statistical Learning> Li Han p171~189
        [2] http://www.tuicool.com/articles/3iENzaV
'''

import numpy as np


class HMM:

    def __init__(self, Ann=[[0]], Bnm=[[0]], pi1n=[0]):
        self.A = np.array(Ann)
        self.B = np.array(Bnm)
        self.pi = np.array(pi1n)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]

    def printhmm(self):
        print "=================================================="
        print "HMM content: N =", self.N, ",M =", self.M
        for i in range(self.N):
            if i == 0:
                print "hmm.A ", self.A[i, :], " hmm.B ", self.B[i, :]
            else:
                print "      ", self.A[i, :], "       ", self.B[i, :]
        print "hmm.pi", self.pi
        print "=================================================="

    def compute_alpha(self, O):
        """
        function: calculate forward prob alpha(t,i), definition can be seen in [1] p175 (10.14)
        :param O: obersive list in 1-D aeeay like O = np.array([0, 1, 0])
        :return alpha: forward prob in 2-D array, alpha[t,i] means alpha_{t}(i)
        """
        # step1: initialization
        T = len(O)
        alpha = np.zeros((T, self.N), np.float)
        for i in range(self.N):
            alpha[0, i] = self.pi[i] * self.B[i, O[0]]

        # step2: induction
        for t in range(T - 1):
            for j in range(self.N):
                sum_alpha = 0.0
                for i in range(self.N):
                    sum_alpha += alpha[t, i] * self.A[i, j]
                alpha[t + 1, j] = sum_alpha * self.B[j, O[t + 1]]

        return alpha

    def compute_beta(self, O):
        """
        function: calculate forward prob alpha(t,i), definition can be seen in [1] p178 (10.18)
        :param O: obersive list in 1-D aeeay like O = np.array([0, 1, 0])
        :return beta: beta in 2_D array, beta[t, i] means gamma_{t}(i)
        """
        # step1: initalization
        T = len(O)
        beta = np.zeros((T, self.N), np.float)
        for i in range(self.N):
            beta[T - 1, i] = 1.0

        # step2: induction
        for t in range(T - 2,-1,-1):
            for i in range(self.N):
                sum_beta = 0.0
                for j in range(self.N):
                    sum_beta += self.A[i, j] * self.B[j, O[t + 1]] * beta[t + 1, j]
                beta[t, i] = sum_beta

        return beta

    def compute_gamma(self, O):
        """
        function: calculate gamma, definition can be seen in [1] p179 (10.23)
        :param O: obersive list in 1-D aeeay like O = np.array([0, 1, 0])
        :return gamma: gamma in 2-D array, gamma[t,i] means gamma_{t}(i)
        """
        T = len(O)
        gamma = np.zeros((T, self.N), np.float)
        alpha = self.compute_alpha(O)
        beta = self.compute_beta(O)
        for t in range(T):
            for i in range(self.N):
                sum_N = 0.0
                for j in range(self.N):
                    sum_N += alpha[t, j] * beta[t, j]
                gamma[t, i] = (alpha[t, i] * beta[t, i]) / sum_N
        return gamma

    def compute_xi(self, O):
        """
        function: calculate xi, definition can be seen in [1] p179 (10.25)
        :param O: obersive list in 1-D aeeay like O = np.array([0, 1, 0])
        :return xi: xi in 3-D array, xi[t,i,j] means xi_{t}(i,j)
        """
        T = len(O)
        xi = np.zeros((T - 1, self.N, self.N))
        alpha = self.compute_alpha(O)
        beta = self.compute_beta(O)

        for t in range(T - 1):
            sum_NN = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    sum_NN = alpha[t, i] * self.A[i, j] * self.B[j, O[t + 1]] * beta[t + 1, j]
            for i in range(self.N):
                for j in range(self.N):
                    xi[t, i, j] = (alpha[t, i] * self.A[i, j] * self.B[j, O[t + 1]] * beta[t + 1, j])/sum_NN
        return xi

    def forward(self, O):
        """
        function: forward algorithm to calculate P(O|lamda) with the input obersive list O
        :param O: obersive list in 1-D aeeay like O = np.array([0, 1, 0])
        :return pprob: P(O|lamda)
        """
        alpha = self.compute_alpha(O)
        T = len(O)
        sum_N = 0.0
        for i in range(self.N):
            sum_N += alpha[T - 1, i]
        pprob = sum_N
        return pprob

    def backward(self, O):
        """
        function: backward algorithm to calculate P(O|lamda) with the input obersive list O
        :param O: obersive list in 1-D aeeay like O = np.array([0, 1, 0])
        :return pprob: P(O|lamda)
        """
        beta = self.compute_beta(O)
        sum_N = 0.0
        for i in range(self.N):
            sum_N += self.pi[i] * self.B[i, O[0]] * beta[0, i]
        pprob = sum_N
        return pprob

    def viterbi(self, O):
        """
        function: used for predict problem, and can be seen in [1] p184
        :param O: obersive list in 1-D aeeay like O = np.array([0, 1, 0])
        :return I: state list of the most possibility
                prob: possibility
        """
        T = len(O)
        # initial
        delta = np.zeros((T, self.N), np.float)
        phi = np.zeros((T, self.N), np.float)
        I = np.zeros(T)
        for i in range(self.N):
            delta[0, i] = self.pi[i] * self.B[i, O[0]]
            phi[0, i] = 0
        # induction
        for t in range(1, T):
            for i in range(self.N):
                delta[t,i] = self.B[i, O[t]] * np.array([delta[t - 1, j] * self.A[j, i] for j in range(self.N)]).max()
                phi[t, i] = np.array([delta[t - 1,j] * self.A[j, i] for j in range(self.N)]).argmax()
        # terminal
        prob = delta[T - 1, :].max()
        I[T - 1] = delta[T - 1, :].argmax()
        # get I
        for t in range(T - 2, -1, -1):
            I[t] = phi[t + 1, I[t + 1]]
        return I, prob

    def baum_welch(self, O, num_observed_value=-1, num_state=-1, num_itera=1000):
        """
        function: baum-welch method (so called EM algorithm) is to train patameters (A, B, pi) in HMM model
                  by a set of obersive list O, which is a unsupervied learning. reference [1] in (10.38)
        :param num_observed_value:
        :param num_state:
        :param O_set: a set of obersive list O, 2-D array like O_set = np.array([[0,1,0],[1,1,0],[1,1,1]])
        :output: (A, B, pi) in HMM model
        """
        print " baum_welch function has some problem, please don't use now!!!!!"
        raise ValueError
        # step1: initial
        self.A = np.ones((num_state, num_state), np.float)
        self.B = np.ones((num_state, num_observed_value), np.float)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]
        self.pi = np.ones((num_state,), np.float)
        self.A = self.A / self.N
        self.B = self.B / self.M
        self.pi = self.pi / self.N
        T = len(O)

        # step2: induction
        for n in range(num_itera):
            xi = self.compute_xi(O)
            gamma = self.compute_gamma(O)
            sum_xi = np.sum(xi, axis=0)
            sum_gamma = np.sum(gamma, axis=0)

            # calculate A[i, j] (n+1)
            for i in range(self.N):
                for j in range(self.N):
                    self.A[i, j] = sum_xi[i, j] / (sum_gamma[i] - gamma[-1, j])

            # calculate B[i,j] (n+1)
            for i in range(self.N):
                for j in range(self.M):
                    sum_gamma_j = 0.0
                    for t in range(T):
                        if O[t] == j:
                            sum_gamma_j += gamma[t, j]
                    self.B[i, j] = sum_gamma_j / sum_gamma[i]

            # calculate pi
            self.pi = gamma[1, :]

    def train(self, I, O, num_state, num_obserivation, init_observation=-1, init_state=-1):
        """
        function: training HMM
        :param I: state list like I = np.array([[0,1,2],[1,0,1],[1,2,0],])
        :param O: observation list like O =     O = np.array([[0,1,1],[1,0,1],[1,1,0],])
        :param num_state: the number of state, lke 3
        :param num_obserivation: the number of observation, like 2
        :param init_observation: the index of init observation, like 1
        :param init_state: the index of init starw, like 2
        """
        print "statr training HMM..."
        self.N = num_state
        self.M = num_obserivation

        # count num_A[i,j] standing for the numbers of state i translating to state j
        num_A = np.zeros((num_state, num_state), np.float)
        for i in range(self.N):
            for j in range(self.N):
                num_i2j = 0
                for i_I in range(I.shape[0]):
                    for j_I in range(I.shape[1] - 1):
                        if I[i_I, j_I] == i and I[i_I, j_I + 1] == j:
                            num_i2j += 1
                num_A[i, j] = num_i2j

        # count num_B[i,j] standing for the numbers of state i translating to obsrtvation j
        num_B = np.zeros((num_state, num_obserivation), np.float)
        for i in range(self.N):
            for j in range(self.M):
                num_i2j = 0
                for i_I in range(I.shape[0]):
                    for j_I in range(I.shape[1]):
                        if I[i_I, j_I] == i and O[i_I, j_I] == j:
                            num_i2j += 1
                num_B[i, j] = num_i2j

        self.A = num_A / np.sum(np.mat(num_A), axis=1).A
        self.B = num_B / np.sum(np.mat(num_B), axis=1).A

        # calculate pi according init_observation or init_state
        if init_state != -1:
            print "init pi with init_state!"
            pi_temp = np.zeros((self.N,), np.float)
            self.pi = pi_temp[init_state] = 1.0
        elif init_observation != -1:
            print "init pi with init_observation!"
            self.pi = self.B[:, init_observation] / np.sum(self.B[:, init_observation])
        else:
            print "init pi with state list I!"
            self.pi = np.zeros((self.N,), np.float)
            for i in range(self.N):
                num_state_i = 0
                for line in I:
                    if line[0] == i:
                        num_state_i += 1
                self.pi[i] = num_state_i
            self.pi = self.pi/np.sum(self.pi, axis=0)

        print "finished train successfully! the hmm is:"
        self.printhmm()




if __name__ == "__main__":
    # 已知hmm模型, 用来预测
    print "python my HMM"
    A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5],
    ])
    B = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3],
    ])
    pi = np.array([0.2, 0.4, 0.4])
    hmm = HMM(A, B, pi)
    O1 = np.array([0, 0, 0])
    hmm.printhmm()
    print hmm.viterbi(O1)
    print hmm.forward(O1)

    # 已知观测序列与对应的状态序列, 训练得到hmm模型
    I = np.array([
        [0,1,2],
        [1,0,1],
        [1,2,0],
    ])
    O = np.array([
        [0,1,1],
        [1,0,1],
        [1,1,0],
    ])
    hmm2 = HMM()
    hmm2.train(I, O, 3, 2)  # 未知初始状态或观测值
    hmm2.train(I, O, 3, 2, init_observation=0)  # 已知初开始观测值
    hmm2.train(I, O, 3, 2, init_state=0)  # 已知初始状态


