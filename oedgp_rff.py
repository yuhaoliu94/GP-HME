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
from odgp_rff import OdgpRff

current_milli_time = lambda: int(round(time.time() * 1000))


class OedgpRff(object):

    def __init__(self, dataset, likelihood_fun, num_examples, d_in, d_out, n_layers, n_rff, df, kernel_type, n_candidates, VI = True):
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
        :param n_candidates: The number of candidate models to ensemble
        :param VI: Whether we use the variational inference as prior or use naive prior
        """
        self.likelihood = likelihood_fun
        self.kernel_type = kernel_type
        self.VI = VI
        self.dataset = dataset

        ## These are all scalars
        self.num_examples = num_examples
        self.nl = n_layers  ## Number of hidden layers
        self.nc = n_candidates  ## Number of candidate models
        
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

        self.T = max(self.dhat_in)

        ## Initialize lists of ensemble models
        self.ensemble = [OdgpRff(dataset, likelihood_fun, num_examples, d_in, d_out, n_layers, n_rff, df, kernel_type, i, VI) for i in range(n_candidates)]
        self.weight = np.array([1 / n_candidates for _ in range(n_candidates)])
        self.pred = [None for _ in range(n_candidates)]


    def learn(self, mc_train, n_iterations=None, display_step=100,  duration = None, test=None, mc_test=1, loss_function=None, less_prints=False):
        print(">>> Ensemble Online learning starts.")

        if n_iterations == 0:
            n_iterations = self.num_examples - max(self.dhat_in)

        mnll_train = 0

        ## Present data to DGP n_iterations times
        for iteration in range(n_iterations):

            total_train_time = 0

            ## Present candidate models to the DGP
            ells = []
            for i in range(self.nc):
                if self.weight[i] > 1e-16:
                    self.ensemble[i].learn(mc_train, n_iterations, display_step, test, mc_test, loss_function)
                    total_train_time += self.ensemble[i].total_train_time
                    ells.append(self.ensemble[i].ell)
                else:
                    self.weight[i] = 0
                    ells.append(0)

            ell_mean = np.average(ells, weights=self.weight)
            mnll_train = (mnll_train * iteration + ell_mean) / (iteration + 1)

            self.T += 1

            ## Display logs every "FLAGS.display_step" iterations
            if iteration % display_step == 0 or iteration == n_iterations - 1:
                self.T -= 1

                start_predict_time = current_milli_time()

                if less_prints:
                    print(">>> i=" + repr(iteration) + "  n=" + repr(self.T // self.num_examples) , end = " ")

                else:
                    print(">>> i=" + repr(iteration) + "  n=" + repr(self.T // self.num_examples) + "  mnll_train=" + repr(-mnll_train) , end="  ")

                if loss_function is not None:

                    for i in range(self.nc):
                        self.pred[i] = self.ensemble[i].pred
                    pred = np.average(np.array(self.pred), axis=0, weights=self.weight)

                    mnll_test = [self.ensemble[i].mnll_test for i in range(self.nc)]
                    mnll_test = np.average(mnll_test, weights=self.weight)

                    elapsed_time = total_train_time + (current_milli_time() - start_predict_time)

                    loss_eval = loss_function.eval(test.Y, pred)
                    print(loss_function.get_name() + "=" + "%.4f" % loss_eval, end=" ")
                    print(" nll_test=" + "%.5f" % (-mnll_test), end=" ")

                    print(" time=" + repr(elapsed_time), end=" ")
                    print("")

                self.T += 1

            for i in range(self.nc):
                self.weight[i] *= np.exp(self.ensemble[i].ell)

            self.weight /= sum(self.weight)

            ## Stop after a given budget of minutes is reached
            if (total_train_time > 1000 * 60 * duration):
                break

        print(">>> Ensemble Online learning ends.")
