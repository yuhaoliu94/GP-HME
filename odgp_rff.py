from __future__ import print_function

import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import DataSet
import utils
import likelihoods
import time

current_milli_time = lambda: int(round(time.time() * 1000))


class OdgpRff(object):
    def __init__(self, dataset, likelihood_fun, num_examples, d_in, d_out, n_layers, n_rff, df, kernel_type, index, VI = True):
        """
        :param dataset: The training dataset including X and Y
        :param likelihood_fun: Likelihood function
        :param num_examples: total number of input samples
        :param d_in: Dimensionality of the input
        :param d_out: Dimensionality of the output
        :param n_layers: Number of hidden layers
        :param n_rff: Number of random features for each layer
        :param df: Number of GPs for each layer
        :param kernel_type: Kernel type: currently only random Fourier features for RBF and arccosine kernels are implemented
        :param index: The index of candidate model
        :param VI: Whether use variatioanl inference as prior or use naive prior
        """
        self.hidden_likelihood = likelihoods.Student_t()
        self.prior_likelihood = likelihoods.Gaussian()
        self.likelihood = likelihood_fun
        self.kernel_type = kernel_type
        self.VI = VI
        self.dataset = dataset

        ## These are all scalars
        self.num_examples = num_examples
        self.nl = n_layers  ## Number of hidden layers
        self.n_Omega = n_layers  ## Number of weigh matrices is "Number of hidden layers"
        self.n_W = n_layers
        self.index = index
        self.total_train_time = 0
            
        ## These are arrays to allow flexibility in the future
        self.n_rff = n_rff * np.ones(n_layers, dtype=np.int32)
        self.df = df * np.ones(n_layers - 1, dtype=np.int32)

        ## Dimensionality of Omega matrices
        self.d_in = np.concatenate([[d_in], self.df])
        self.d_out = self.n_rff

        ## Dimensionality of W matrices
        if self.kernel_type == "RBF":
            self.dhat_in = self.n_rff * 2
            self.dhat_out = np.concatenate([self.df, [d_out]])

        if self.kernel_type == "arccosine":
            self.dhat_in = self.n_rff
            self.dhat_out = np.concatenate([self.df, [d_out]])

        if VI:
            self.T = max(self.dhat_in)
        else:
            self.T = 0

        ## Initialize posterior parameters
        self.log_theta_sigma2 = self.init_prior_log_theta_sigma2()

        self.Omega = self.init_prior_Omega()

        self.mean_W, self.Sigma_W = self.init_prior_W()

        self.Phi = self.init_prior_Phi()

        self.r = self.init_prior_r()

    ## Function to initialize the posterior over Omega
    def init_prior_Omega(self):
        if self.VI:
            Omega = []
            for i in range(self.n_Omega):
                Omega_VI = np.loadtxt("Initialization/" + self.dataset.name + "/Omega_" + str(i) + "_" + str(self.index))
                Omega.append(np.atleast_2d(Omega_VI))
        else:
            Omega = []
            for i in range(self.n_Omega):
                Lambda = 1 / utils.get_random(size=self.d_in[i])
                Omega_Naive = utils.get_mvn_samples(mean=np.zeros(self.d_in[i]), cov=np.diag(Lambda), shape=self.d_out[i])
                Omega.append(np.atleast_2d(Omega_Naive).T)

        return Omega

    ## Function to initialize the posterior over W
    def init_prior_W(self):
        if self.VI:
            mean_W = []
            Sigma_W = []
            for i in range(self.n_W):
                mean_W_VI = np.loadtxt("Initialization/" + self.dataset.name + "/mean_W_" + str(i))
                if mean_W_VI.ndim == 1:
                    mean_W_VI = np.atleast_2d(mean_W_VI).T
                mean_W.append(mean_W_VI.T * np.exp(0.5 * self.log_theta_sigma2[i]))

                Sigma_W.append(np.zeros([self.dhat_out[i], self.dhat_in[i], self.dhat_in[i]]))
                log_var_W_VI = np.loadtxt("Initialization/" + self.dataset.name + "/log_var_W_" + str(i))
                if log_var_W_VI.ndim == 1:
                    var_W_VI = np.exp(np.atleast_2d(log_var_W_VI).T)
                else:
                    var_W_VI = np.exp(np.atleast_2d(log_var_W_VI))
                for j in range(self.dhat_out[i]):
                    Sigma_W_VI = np.diag(var_W_VI[:,j]) * np.exp(self.log_theta_sigma2[i])
                    Sigma_W[-1][j] = np.atleast_2d(Sigma_W_VI)
        else:
            mean_W = [np.zeros([self.dhat_out[i], self.dhat_in[i]]) for i in range(self.n_W)]
            Sigma_W = [np.zeros([self.dhat_out[i], self.dhat_in[i], self.dhat_in[i]]) for i in range(self.n_W)]
            for i in range(self.n_W):
                for j in range(self.dhat_out[i]):
                    Sigma_W[i][j,:,:] = np.eye(self.dhat_in[i]) * np.exp(self.log_theta_sigma2[i])

        return mean_W, Sigma_W

    ## Function to initialize the posterior over log_theta_sigma2
    def init_prior_log_theta_sigma2(self):
        if self.VI:
            log_theta_sigma2 = np.loadtxt("Initialization/" + self.dataset.name + "/log_theta_sigma2")
        else:
            log_theta_sigma2 = np.ones(self.n_Omega) # utils.get_random(self.n_Omega)

        return log_theta_sigma2.ravel()

    ## Function to initialize the posterior over hidden layers
    def init_prior_Phi(self):
        N_prior = max(self.dhat_in)
        if self.VI:
            Phi = []
            for i in range(1, self.nl):
                Phi_VI = np.loadtxt("Initialization/" + self.dataset.name + "/hidden_layer_" + str(i) + "_" + str(self.index))
                Phi.append(np.atleast_2d(Phi_VI).T)
            last_layer = np.atleast_2d(self.dataset.Y[:N_prior,:])
            Phi.append(last_layer.T)
        else:
            Phi = [np.zeros([self.dhat_out[i-1], N_prior]) for i in range(1, self.nl)]

            self.T_prior = 0

            # print(">>> Initialization start.")

            mnll_prior = 0

            for iteration in range(N_prior * 1):
                start_train_time = current_milli_time()

                ell, _ = self.forward_prior()
                mnll_prior = (mnll_prior * iteration + ell) / (iteration + 1)

                # if iteration % 250 == 0:
                #     print(">>> i=" + repr(iteration) + "  n=" + repr(self.T_prior // N_prior) + "  mnll_train=" + repr(-mnll_prior) , end="  ")
                #     print("")

                self.backward_prior()
                for i in range(1, self.nl):
                    Phi[i-1][:, self.T_prior % N_prior] = self.backward_layer[i]

                self.update_prior()

                self.T_prior += 1

                self.total_train_time += current_milli_time() - start_train_time
            last_layer = np.atleast_2d(self.dataset.Y[:N_prior, :])
            Phi.append(last_layer.T)

            self.T = self.T_prior

        return Phi

    def forward_prior(self, mc=100):

        self.noise = self.sample_from_noise(mc, prior=True)

        Din = self.d_in[0]
        x = self.dataset.X[self.T_prior % max(self.dhat_in),:]
        y = self.dataset.Y[self.T_prior % max(self.dhat_in),:]

        ## The representation of the information is based on 2-dimensional ndarrays (one for each layer)
        ## Each slice [i,:] of these ndarrays is one Monte Carlo realization of the value of the hidden units
        ## At layer zero we simply replicate the input matrix X self.mc times
        self.forward_layer = []
        self.forward_mean = []
        self.forward_var = []

        self.forward_layer.append(np.ones([mc, Din]) * x)
        self.forward_mean.append(np.ones([mc, Din]) * x)
        self.forward_var.append(np.zeros([mc, Din]))

        ## Forward propagate information from the input to the output through hidden layers
        for i in range(self.nl):
            layer_times_Omega = np.dot(self.forward_layer[i], self.Omega[i])  # x * Omega

            ## Apply the activation function corresponding to the chosen kernel - PHI
            if self.kernel_type == "RBF":
                Phi = 1 / np.sqrt(self.n_rff[i]) * np.concatenate([np.cos(layer_times_Omega), np.sin(layer_times_Omega)], axis=1)
            elif self.kernel_type == "arccosine":
                Phi = 1 / np.sqrt(self.n_rff[i]) * np.maximum(layer_times_Omega, 0.0)

            mean = np.dot(Phi, self.mean_W[i].T)  # mc * dhat_out
            self.forward_mean.append(mean)

            var = np.zeros([mc, self.dhat_out[i]])
            for j in range(self.dhat_out[i]):
                var[:,j] = utils.diag(Phi, self.Sigma_W[i][j,:,:]) + 1e-4
            self.forward_var.append(var)

            F = mean + np.sqrt(var) * self.noise[i]
            self.forward_layer.append(F)

        ## Output layer
        mean_out = self.forward_mean[self.nl]
        var_out = self.forward_var[self.nl]

        ## Given the output layer, we compute the conditional likelihood across all samples
        if self.likelihood.get_name() == "Classification":
            ll = self.likelihood.log_cond_prob(y, mean_out)
            ell = np.mean(ll)

        elif self.likelihood.get_name() == "Regression":
            ll = self.prior_likelihood.log_cond_prob(y, mean_out, np.sqrt(var_out))
            ## Mini-batch estimation of the expected log-likelihood term
            ell = sum(np.average(ll,0))

        return ell, mean_out

    def backward_prior(self):

        x = self.dataset.X[self.T_prior % max(self.dhat_in), :]
        y = self.dataset.Y[self.T_prior % max(self.dhat_in), :]

        ## The representation of the information is based on 2-dimensional ndarrays (one for each layer)
        ## Each slice [i,:] of these tensors is one Monte Carlo realization of the value of the hidden units
        ## At layer zero we simply replicate the input matrix X self.mc times
        self.backward_layer = []

        self.backward_layer.insert(0, y)

        ## Backward propagate information from the output to the input through hidden layers
        for i in reversed(range(2, self.nl+1)):
            if i == self.nl and self.likelihood.get_name() == "Classification":
                log_weights = self.likelihood.log_cond_prob(self.backward_layer[0], self.forward_mean[i])
                log_weights = log_weights.reshape(-1,1)
            else:
                log_weights = self.prior_likelihood.log_cond_prob(self.backward_layer[0], self.forward_mean[i], np.sqrt(self.forward_var[i]))
            log_weight = np.sum(log_weights, axis=1)
            weight = utils.normalize_weight(log_weight)

            F = np.average(self.forward_layer[i - 1], axis=0, weights = weight)
            self.backward_layer.insert(0, F)

        self.backward_layer.insert(0, x)

    def update_prior(self):
        for i in range(self.nl):
            layer_times_Omega = np.dot(self.backward_layer[i], self.Omega[i])  # n_rff

            ## Apply the activation function corresponding to the chosen kernel - PHI
            if self.kernel_type == "RBF":
                phi = 1 / np.sqrt(self.n_rff[i]) * np.concatenate([np.cos(layer_times_Omega), np.sin(layer_times_Omega)], axis=0)
            elif self.kernel_type == "arccosine":
                phi = 1 / np.sqrt(self.n_rff[i]) * np.maximum(layer_times_Omega,0.0)

            mean = np.dot(phi, self.mean_W[i].T)
            residual = self.backward_layer[i + 1] - mean
            for j in range(self.dhat_out[i]):
                tmp = np.dot(phi, self.Sigma_W[i][j, :, :])
                sufficient = 1e-4 + np.dot(tmp, phi)
                k = tmp / sufficient
                self.mean_W[i][j,:] += k * residual[j]
                self.Sigma_W[i][j,:,:] = np.dot(np.eye(self.dhat_in[i]) - np.dot(k.reshape(-1,1), phi.reshape(1,-1)), self.Sigma_W[i][j,:,:])

    ## Function to initialize the auxiliary parameter r
    def init_prior_r(self):
        r = []
        for i in range(self.nl):
            r_i_ = np.zeros(self.dhat_out[i])
            for j in range(self.dhat_out[i]):
                hidden_layer = self.Phi[i][j,:]
                mean_W = self.mean_W[i][j,:]
                Sigma_W = self.Sigma_W[i][j,:,:]
                Sigma_W_inverse = np.linalg.inv(Sigma_W + 1e-8 * np.eye(self.dhat_in[i]))
                r_i_[j] = max(np.dot(hidden_layer, hidden_layer) - np.dot(np.dot(mean_W, Sigma_W_inverse), mean_W), 1e-4)
            r.append(r_i_)

        return r

    ## Returns Student t's noises for layers
    def sample_from_noise(self, mc, task = "training", prior = False):
        noise = []
        for i in range(self.nl):
            if task == "training":
                if prior:
                    z = utils.get_normal_samples([mc, self.dhat_out[i]])
                else:
                    nu = self.T - self.dhat_in[i] + 1
                    z = utils.get_t_samples([mc, self.dhat_out[i]], nu)
            else:
                z = np.zeros([mc, self.dhat_out[i]])
            noise.append(z)

        return noise

    ## Returns the expected log-likelihood term and prediction
    def forward(self, mc=1, task="training", x=None, y=None):
        if task == "training":
            # if self.T % max(self.dhat_in) == 0:
            self.noise = self.sample_from_noise(mc, task)
            noise = self.noise
        elif task == "prediction":
            noise = self.sample_from_noise(mc, "training")

        Din = self.d_in[0]
        if x is None:
            x = self.dataset.X[self.T % self.num_examples,:]
            y = self.dataset.Y[self.T % self.num_examples,:]

        ## The representation of the information is based on 2-dimensional ndarrays (one for each layer)
        ## Each slice [i,:] of these ndarrays is one Monte Carlo realization of the value of the hidden units
        ## At layer zero we simply replicate the input matrix X self.mc times
        self.forward_layer = []
        self.forward_mean = []
        self.forward_var = []

        self.forward_layer.append(np.ones([mc, Din]) * x)
        self.forward_mean.append(np.ones([mc, Din]) * x)
        self.forward_var.append(np.zeros([mc, Din]))

        ## Forward propagate information from the input to the output through hidden layers
        for i in range(self.nl):
            nu = self.T - self.dhat_in[i] + 1
            layer_times_Omega = np.dot(self.forward_layer[i], self.Omega[i])  # x * Omega

            ## Apply the activation function corresponding to the chosen kernel - PHI
            if self.kernel_type == "RBF":
                Phi = 1 / np.sqrt(self.n_rff[i]) * np.concatenate([np.cos(layer_times_Omega), np.sin(layer_times_Omega)], axis=1)
            elif self.kernel_type == "arccosine":
                Phi = 1 / np.sqrt(self.n_rff[i]) * np.maximum(layer_times_Omega, 0.0)

            mean = np.dot(Phi, self.mean_W[i].T)  # mc * dhat_out
            self.forward_mean.append(mean)

            a = np.zeros([mc, self.dhat_out[i]])
            for j in range(self.dhat_out[i]):
                a[:,j] = self.r[i][j] * (1 + utils.diag(Phi, self.Sigma_W[i][j,:,:]))
            var = a / nu
            self.forward_var.append(var)

            F = mean + np.sqrt(var) * noise[i]
            self.forward_layer.append(F)

        ## Output layer
        mean_out = self.forward_mean[self.nl]
        var_out = self.forward_var[self.nl]

        ## Given the output layer, we compute the conditional likelihood across all samples
        if self.likelihood.get_name() == "Classification":
            ll = self.likelihood.log_cond_prob(y, mean_out)
            ell = np.mean(ll)

        elif self.likelihood.get_name() == "Regression":
            ll = self.likelihood.log_cond_prob(y, mean_out, np.sqrt(var_out), nu)
            ## Mini-batch estimation of the expected log-likelihood term
            ell = sum(np.average(ll,0))

        return ell, mean_out

    def backward(self):

        x = self.dataset.X[self.T % self.num_examples, :]
        y = self.dataset.Y[self.T % self.num_examples, :]
        
        self.backward_layer = []

        self.backward_layer.insert(0, y)

        ## Backrward propagate information from the output to the input through hidden layers
        for i in reversed(range(2, self.nl+1)):
            if i == self.nl and self.likelihood.get_name() == "Classification":
                log_weights = self.likelihood.log_cond_prob(self.backward_layer[0], self.forward_mean[i])
                log_weights = log_weights.reshape(-1,1)
            else:
                nu = self.T - self.dhat_in[i - 1] + 1
                log_weights = self.hidden_likelihood.log_cond_prob(self.backward_layer[0], self.forward_mean[i], np.sqrt(self.forward_var[i]), nu)

            log_weight = np.sum(log_weights, axis=1)
            weight = utils.normalize_weight(log_weight)

            F = np.average(self.forward_layer[i - 1], axis=0, weights = weight)
            self.backward_layer.insert(0, F)

        self.backward_layer.insert(0, x)

    def update(self):
        for i in range(self.nl):
            layer_times_Omega = np.dot(self.backward_layer[i], self.Omega[i])  # n_rff

            ## Apply the activation function corresponding to the chosen kernel - PHI
            if self.kernel_type == "RBF":
                phi = 1 / np.sqrt(self.n_rff[i]) * np.concatenate([np.cos(layer_times_Omega), np.sin(layer_times_Omega)], axis=0)
            elif self.kernel_type == "arccosine":
                phi = 1 / np.sqrt(self.n_rff[i]) * np.maximum(layer_times_Omega,0.0)

            mean = np.dot(phi, self.mean_W[i].T)  # dhat_out
            residual = self.backward_layer[i + 1] - mean
            for j in range(self.dhat_out[i]):
                tmp = np.dot(phi, self.Sigma_W[i][j, :, :])
                sufficient = 1 + np.dot(tmp, phi)
                k = tmp / sufficient
                self.r[i][j] += pow(residual[j], 2) / sufficient
                self.mean_W[i][j,:] += k * residual[j]
                self.Sigma_W[i][j,:,:] = np.dot(np.eye(self.dhat_in[i]) - np.dot(k.reshape(-1,1), phi.reshape(1,-1)), self.Sigma_W[i][j,:,:])

    ## Function that learns the deep GP model sequentially with random Fourier feature approximation
    def learn(self, mc_train, n_iterations=None, display_step=100, test=None, mc_test=1, loss_function=None):

        N_test = test.X.shape[0]

        ## Present one sample to the DGP
        start_train_time = current_milli_time()

        ell, _ = self.forward(mc=mc_train, task="training")
        iteration = self.T - max(self.dhat_in)
        self.ell = ell

        self.backward()

        self.update()

        self.total_train_time += current_milli_time() - start_train_time

        ## Display logs every "FLAGS.display_step" iterations
        if iteration % display_step == 0 or iteration == n_iterations - 1:
            start_predict_time = current_milli_time()

            if loss_function is not None:
                mnll_test = 0
                preds = []
                for i in range(N_test):
                    ell, pred = self.forward(mc=mc_test, task="prediction", x=test.X[i,:], y=test.Y[i,:])
                    mnll_test = (mnll_test * i + ell) / (i + 1)
                    preds.append(self.likelihood.predict(pred))

                self.total_train_time += current_milli_time() - start_predict_time

                self.mnll_test = mnll_test
                self.pred = np.mean(np.array(preds),1)

        self.T += 1
