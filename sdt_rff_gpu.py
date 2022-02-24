from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import DataSet
import utils
import likelihoods
import time
import os

current_milli_time = lambda: int(round(time.time() * 1000))

class SdtRff(object):
    def __init__(self, likelihood_fun, num_examples, d_in, d_out, h_tree, n_rff, kernel_type, ard_type, local_reparam, q_Omega_fixed, theta_fixed, likelihood_type, dataset, fold):
        """
        :param likelihood_fun: Likelihood function
        :param num_examples: total number of input samples
        :param d_in: Dimensionality of the input
        :param d_out: Dimensionality of the output
        :param h_tree: Height of tree
        :param n_rff: Number of random features for each node
        :param kernel_type: RBF, arccosine, or identity kernels
        :param ard_type: Whether the Omega is ard cross the tree
        :param Omega_fixed: Whether the Omega weights should be fixed throughout the optimization
        :param theta_fixed: Whether covariance parameters should be fixed throughout the optimization
        :param local_reparam: Whether to use the local reparameterization trick or not
        """
        self.likelihood = likelihood_fun
        self.likelihood_type = likelihood_type
        self.kernel_type = kernel_type
        self.ard_type = ard_type
        self.q_Omega_fixed = q_Omega_fixed
        self.theta_fixed = theta_fixed
        self.q_Omega_fixed_flag = q_Omega_fixed > 0
        self.theta_fixed_flag = theta_fixed > 0
        self.local_reparam = local_reparam
        self.dataset = dataset
        self.fold = fold

        ## These are all scalars
        self.num_examples = num_examples
        self.h = h_tree ## Height of tree
        self.h_Omega = h_tree  ## Height of weight matrices is "Height of tree"
        self.h_W = h_tree
        self.n_rff = n_rff

        ## Dimensionality of Omega matrices
        self.d_in = d_in
        self.d_out = n_rff

        ## Dimensionality of W matrices
        if self.kernel_type == "RBF":
            self.dhat_in = self.n_rff * 2
            self.dhat_out = np.concatenate([np.ones(h_tree-1, dtype=np.int32), [d_out]])

        if self.kernel_type == "arccosine":
            self.dhat_in = self.n_rff
            self.dhat_out = np.concatenate([np.ones(h_tree-1, dtype=np.int32), [d_out]])
            
        if self.kernel_type == "identity":
            self.dhat_in = d_in + 1
            self.dhat_out = np.concatenate([np.ones(h_tree-1, dtype=np.int32), [d_out]])

        ## When Omega is optimized, fix some standard normals throughout the execution that will be used to construct Omega
        if ard_type == 2:
            self.z_for_Omega_fixed = []
            for i in range(self.h_Omega):
                self.z_for_Omega_fixed.append([])
                for j in range(pow(2,i)):
                    tmp = utils.get_normal_samples(1, self.d_in, self.d_out)
                    self.z_for_Omega_fixed[-1].append(tf.Variable(tmp[0,:,:], trainable = False))
        elif ard_type == 1:
            self.z_for_Omega_fixed = []
            for i in range(self.h_Omega):
                tmp = utils.get_normal_samples(1, self.d_in, self.d_out)
                self.z_for_Omega_fixed.append(tf.Variable(tmp[0, :, :], trainable=False))
        else:
            tmp = utils.get_normal_samples(1, self.d_in, self.d_out)
            self.z_for_Omega_fixed = tf.Variable(tmp[0, :, :], trainable=False)

        ## Parameters defining prior over Omega
        if ard_type == 2:
            self.log_theta_sigma2 = []
            for i in range(self.h_Omega):
                 self.log_theta_sigma2.append(tf.Variable(tf.zeros([pow(2,i)]), name="log_theta_sigma2"))
        elif ard_type == 1:
            self.log_theta_sigma2 = tf.Variable(tf.zeros([self.h_Omega]), name="log_theta_sigma2")
        else:
            self.log_theta_sigma2 = tf.Variable(tf.zeros([1]), name="log_theta_sigma2")

        if ard_type == 2:
            self.llscale0 = []
            for i in range(self.h_Omega):
                self.llscale0.append([])
                for j in range(pow(2,i)):
                    self.llscale0[-1].append(tf.constant(0.5 * np.log(self.d_in), 'float32'))

            self.log_theta_lengthscale = []
            for i in range(self.h_Omega):
                self.log_theta_lengthscale.append([])
                for j in range(pow(2,i)):
                    self.log_theta_lengthscale[-1].append(tf.Variable(tf.multiply(tf.ones([self.d_in]), self.llscale0[i][j]), name="log_theta_lengthscale"))
        elif ard_type == 1:
            self.llscale0 = []
            for i in range(self.h_Omega):
                self.llscale0.append(tf.constant(0.5 * np.log(self.d_in), 'float32'))

            self.log_theta_lengthscale = []
            for i in range(self.h_Omega):
                self.log_theta_lengthscale.append(tf.Variable(tf.multiply(tf.ones([self.d_in]), self.llscale0[i]), name="log_theta_lengthscale"))
        else:
            self.llscale0 = tf.constant(0.5 * np.log(self.d_in), 'float32')
            self.log_theta_lengthscale = tf.Variable(tf.multiply(tf.ones([self.d_in]), self.llscale0), name="log_theta_lengthscale")

        self.prior_mean_Omega, self.log_prior_var_Omega = self.get_prior_Omega(self.log_theta_lengthscale)

        ## Set the prior over weights
        self.prior_mean_W, self.log_prior_var_W = self.get_prior_W()

        ## Initialize posterior parameters
        self.mean_Omega, self.log_var_Omega = self.init_posterior_Omega()

        self.mean_W, self.log_var_W = self.init_posterior_W()

        ## Set the number of Monte Carlo samples as a placeholder so that it can be different for training and test
        self.mc =  tf.placeholder(tf.int32)

        ## Batch data placeholders
        Din = d_in
        Dout = d_out
        self.X = tf.placeholder(tf.float32, [None, Din])
        self.Y = tf.placeholder(tf.float32, [None, Dout])

        ## Builds whole computational graph with relevant quantities as part of the class
        self.loss, self.kl, self.ell, self.Y_out, self.Prob_node = self.get_nelbo()

        ## Initialize the session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ## Definition of a prior for Omega - which depends on the lengthscale of the covariance function
    def get_prior_Omega(self, log_lengthscale):
        prior_mean_Omega = []
        log_prior_var_Omega = []
        if self.ard_type == 2:
            for i in range(self.h_Omega):
                prior_mean_Omega.append([])
                log_prior_var_Omega.append([])
                for j in range(pow(2,i)):
                    prior_mean_Omega[-1].append(tf.zeros([self.d_in,1]))
                    log_prior_var_Omega[-1].append(-2 * log_lengthscale[i][j])
        elif self.ard_type == 1:
            for i in range(self.h_Omega):
                prior_mean_Omega.append(tf.zeros([self.d_in, 1]))
                log_prior_var_Omega.append(-2 * log_lengthscale[i])
        else:
            prior_mean_Omega = tf.zeros([self.d_in, 1])
            log_prior_var_Omega = -2 * log_lengthscale

        return prior_mean_Omega, log_prior_var_Omega

    ## Definition of a prior over W - these are standard normals
    def get_prior_W(self):
        prior_mean_W = tf.zeros(self.h_W)
        log_prior_var_W = tf.zeros(self.h_W)
        return prior_mean_W, log_prior_var_W

    ## Function to initialize the posterior over omega
    def init_posterior_Omega(self):
        mu, sigma2 = self.get_prior_Omega(self.llscale0)
        mean_Omega = []
        log_var_Omega = []
        if self.ard_type == 2:
            for i in range(self.h_Omega):
                mean_Omega.append([])
                log_var_Omega.append([])
                for j in range(pow(2,i)):
                    mean_Omega[-1].append(tf.Variable(mu[i][j] * tf.ones([self.d_in, self.d_out]), name="q_Omega"))
                    log_var_Omega[-1].append(tf.Variable(sigma2[i][j] * tf.ones([self.d_in, self.d_out]), name="q_Omega"))
        elif self.ard_type == 1:
            mean_Omega = [tf.Variable(mu[i] * tf.ones([self.d_in, self.d_out]), name="q_Omega") for i in range(self.h_Omega)]
            log_var_Omega = [tf.Variable(sigma2[i] * tf.ones([self.d_in, self.d_out]), name="q_Omega") for i in range(self.h_Omega)]
        else:
            mean_Omega = tf.Variable(mu * tf.ones([self.d_in, self.d_out]), name="q_Omega")
            log_var_Omega = tf.Variable(sigma2 * tf.ones([self.d_in, self.d_out]), name="q_Omega")
                
        return mean_Omega, log_var_Omega

    ## Function to initialize the posterior over W
    def init_posterior_W(self):
        mean_W = []
        log_var_W = []
        for i in range(self.h_W):
            mean_W.append([])
            log_var_W.append([])
            for j in range(pow(2,i)):
                mean_W[-1].append(tf.Variable(tf.zeros([self.dhat_in, self.dhat_out[i]]), name="q_W"))
                log_var_W[-1].append(tf.Variable(tf.zeros([self.dhat_in, self.dhat_out[i]]), name="q_W"))

        return mean_W, log_var_W

    ## Function to compute the KL divergence between priors and approximate posteriors over model parameters (Omega and W) when q(Omega) is to be learned
    def get_kl(self):
        kl = 0
        if self.ard_type == 2:
            for i in range(self.h_Omega):
                for j in range(pow(2, i)):
                    kl = kl + utils.DKL_gaussian(self.mean_Omega[i][j], self.log_var_Omega[i][j], self.prior_mean_Omega[i][j], self.log_prior_var_Omega[i][j])
        elif self.ard_type == 1:
            for i in range(self.h_Omega):
                kl = kl + utils.DKL_gaussian(self.mean_Omega[i], self.log_var_Omega[i], self.prior_mean_Omega[i], self.log_prior_var_Omega[i])
        else:
            kl = kl + utils.DKL_gaussian(self.mean_Omega, self.log_var_Omega, self.prior_mean_Omega, self.log_prior_var_Omega)
        for i in range(self.h_W):
            for j in range(pow(2, i)):
                kl = kl + utils.DKL_gaussian(self.mean_W[i][j], self.log_var_W[i][j], self.prior_mean_W[i], self.log_prior_var_W[i])

        return kl

    ## Returns Omega values calculated from fixed random variables and mean and variance of q() - the latter are optimized and enter the calculation of the KL so also lengthscale parameters get optimized
    def sample_from_Omega(self):
        Omega_from_q = []
        if self.ard_type == 2:
            for i in range(self.h_Omega):
                Omega_from_q.append([])
                for j in range(pow(2, i)):
                    z = tf.multiply(self.z_for_Omega_fixed[i][j], tf.ones([self.mc, self.d_in, self.d_out]))
                    Omega_from_q[-1].append(tf.add(tf.multiply(z, tf.exp(self.log_var_Omega[i][j] / 2)), self.mean_Omega[i][j]))
        elif self.ard_type == 1:
            for i in range(self.h_Omega):
                z = tf.multiply(self.z_for_Omega_fixed[i], tf.ones([self.mc, self.d_in, self.d_out]))
                Omega_from_q.append(tf.add(tf.multiply(z, tf.exp(self.log_var_Omega[i] / 2)), self.mean_Omega[i]))
        else:
            z = tf.multiply(self.z_for_Omega_fixed, tf.ones([self.mc, self.d_in, self.d_out]))
            Omega_from_q = tf.add(tf.multiply(z, tf.exp(self.log_var_Omega / 2)), self.mean_Omega)

        return Omega_from_q

    ## Returns samples from approximate posterior over W
    def sample_from_W(self):
        W_from_q = []
        for i in range(self.h_W):
            W_from_q.append([])
            for j in range(pow(2, i)):
                z = utils.get_normal_samples(self.mc, self.dhat_in, self.dhat_out[i])
                self.z = z
                W_from_q[-1].append(tf.add(tf.multiply(z, tf.exp(self.log_var_W[i][j] / 2)), self.mean_W[i][j]))
                
        return W_from_q

    ## Returns the expected log-likelihood term in the variational lower bound
    def get_ell(self):
        Din = self.d_in
        MC = self.mc
        h = self.h
        X = self.X
        Y = self.Y
        batch_size = tf.shape(X)[0] # This is the actual batch size when X is passed to the graph of computations

        ## The representation of the information is based on 3-dimensional tensors (one for each layer)
        ## Each slice [i,:,:] of these tensors is one Monte Carlo realization of the value of the hidden units
        ## At input space we simply replicate the input matrix X self.mc times
        self.X_MC = tf.multiply(tf.ones([self.mc, batch_size, Din]), X)

        ## Forward propagate information from the input to the output through the tree
        Omega_from_q  = self.sample_from_Omega()
        
        if not self.local_reparam:
            W_from_q = self.sample_from_W()
        
        ## Probability to the right child
        Prob_node = []
        for i in range(h):
            Prob_node.append([])
            for j in range(pow(2, i)):
                if self.ard_type == 2:
                    X_MC_times_Omega = tf.matmul(self.X_MC, Omega_from_q[i][j])  # X * Omega

                    ## Apply the activation function corresponding to the chosen kernel - PHI
                    if self.kernel_type == "RBF":
                        Phi = tf.exp(0.5 * self.log_theta_sigma2[i][j]) / tf.cast(tf.sqrt(1. * self.n_rff), 'float32') * tf.concat(values=[tf.cos(X_MC_times_Omega), tf.sin(X_MC_times_Omega)], axis=2)
                    if self.kernel_type == "arccosine":
                        Phi = tf.exp(0.5 * self.log_theta_sigma2[i][j]) / tf.cast(tf.sqrt(1. * self.n_rff), 'float32') * tf.concat(values=[tf.maximum(X_MC_times_Omega, 0.0)], axis=2)
                        
                    if self.kernel_type == "identity":
                        Phi = tf.concat(values=[tf.ones([self.mc, batch_size, 1]),self.X_MC], axis=2)
                        
                elif self.ard_type == 1:
                    X_MC_times_Omega = tf.matmul(self.X_MC, Omega_from_q[i])  # X * Omega

                    ## Apply the activation function corresponding to the chosen kernel - PHI
                    if self.kernel_type == "RBF":
                        Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / tf.cast(tf.sqrt(1. * self.n_rff), 'float32') * tf.concat(values=[tf.cos(X_MC_times_Omega), tf.sin(X_MC_times_Omega)], axis=2)
                    if self.kernel_type == "arccosine":
                        Phi = tf.exp(0.5 * self.log_theta_sigma2[i]) / tf.cast(tf.sqrt(1. * self.n_rff), 'float32') * tf.concat(values=[tf.maximum(X_MC_times_Omega, 0.0)], axis=2)
                    if self.kernel_type == "identity":
                        Phi = tf.concat(values=[tf.ones([self.mc, batch_size, 1]),self.X_MC], axis=2)
                        
                else:
                    X_MC_times_Omega = tf.matmul(self.X_MC, Omega_from_q)  # X * Omega

                    ## Apply the activation function corresponding to the chosen kernel - PHI
                    if self.kernel_type == "RBF":
                        Phi = tf.exp(0.5 * self.log_theta_sigma2) / tf.cast(tf.sqrt(1. * self.n_rff), 'float32') * tf.concat(values=[tf.cos(X_MC_times_Omega), tf.sin(X_MC_times_Omega)], axis=2)
                    if self.kernel_type == "arccosine":
                        Phi = tf.exp(0.5 * self.log_theta_sigma2) / tf.cast(tf.sqrt(1. * self.n_rff), 'float32') * tf.concat(values=[tf.maximum(X_MC_times_Omega, 0.0)], axis=2)
                    if self.kernel_type == "identity":
                        Phi = tf.concat(values=[tf.ones([self.mc, batch_size, 1]),self.X_MC], axis=2)

                if self.local_reparam:
                    z_for_F_sample = utils.get_normal_samples(self.mc, tf.shape(Phi)[1], self.dhat_out[i])
                    mean_F = tf.tensordot(Phi, self.mean_W[i][j], [[2], [0]])
                    var_F = tf.tensordot(tf.pow(Phi,2), tf.exp(self.log_var_W[i][j]), [[2],[0]])
                    F = tf.add(tf.multiply(z_for_F_sample, tf.sqrt(var_F)), mean_F)
                else:
                    F = tf.matmul(Phi, W_from_q[i][j]) # (mc, batch_size, 1 or K)
                
                if i != h - 1:
                    Prob_node[-1].append(tf.sigmoid(F))
                else:
                    Prob_node[-1].append(F)
                        
        ## Cumulative probability for nodes
        Cumulative_Prob = [[tf.cast(1, 'float32')]]
        for i in range(1, h):
            Cumulative_Prob.append([])
            for j in range(pow(2, i)):
                parent, direct = divmod(j, 2)
                if direct == 1:
                    Cumulative_Prob[-1].append(tf.multiply(Prob_node[i-1][parent], Cumulative_Prob[i-1][parent]))
                else:
                    Cumulative_Prob[-1].append(tf.multiply(1 - Prob_node[i-1][parent], Cumulative_Prob[i-1][parent]))        
        
        if self.likelihood.get_task() == "Classification":
            Y_out = tf.add_n([tf.multiply(Cumulative_Prob[-1][j], tf.log(tf.nn.softmax(Prob_node[-1][j]))) for j in range(pow(2, h - 1))])
            
            if self.likelihood_type == "standard":
                ## Given the leaves, we compute the log likelihood across all samples
                ll = tf.add_n([tf.multiply(Cumulative_Prob[-1][j][:,:,0], self.likelihood.log_cond_prob(Y, Prob_node[-1][j])) for j in range(pow(2, h-1))])
                
            elif self.likelihood_type == "relative":
                ## Given the leaves, we compute the relative log likelihood across all samples
                ll = self.likelihood.log_cond_prob(Y, Y_out)

            ## Mini-batch estimation of the expected log-likelihood term
            ell = tf.reduce_sum(tf.reduce_mean(ll, 0)) * self.num_examples / tf.cast(batch_size, "float32")
        
        elif self.likelihood.get_task() == "Regression":
            Y_out = tf.add_n([tf.multiply(Cumulative_Prob[-1][j], Prob_node[-1][j]) for j in range(pow(2, h-1))])
            
            ## Given the leaves, we compute the log likelihood across all samples
            ll = tf.add_n([tf.multiply(Cumulative_Prob[-1][j], self.likelihood.log_cond_prob(Y, Prob_node[-1][j])) for j in range(pow(2, h-1))])
            
            ## Mini-batch estimation of the expected log-likelihood term
            ell = tf.reduce_sum(tf.reduce_mean(ll, 0)) * self.num_examples / tf.cast(batch_size, "float32")
            
        ## Regularizer
        C = tf.cast(0, 'float32')
        for i in range(h-1):
            for j in range(pow(2, i)):
                a = tf.reduce_sum(tf.multiply(Prob_node[i][j], Cumulative_Prob[i][j]),1)
                if i == 0:
                    b = tf.cast(batch_size, 'float32')
                else:
                    b = tf.reduce_sum(Cumulative_Prob[i][j],1)
                alpha = tf.reduce_sum(tf.reduce_mean(tf.divide(a,b), 0))
                node_loss = tf.log(alpha) + tf.log(1-alpha)
                C = C + tf.multiply(tf.cast(self.num_examples*pow(2,-i-1), 'float32'), node_loss)

        return ell + C, Y_out, Prob_node

    ## Maximize variational lower bound --> minimize Nelbo
    def get_nelbo(self):
        kl = self.get_kl()
        ell, Y_out, Prob_node = self.get_ell()
        nelbo  = kl - ell
        return nelbo, kl, ell, Y_out, Prob_node

    ## Return predictions on some data
    def predict(self, data, mc_test):
        out = self.likelihood.predict(self.Y_out)

        nll = - tf.reduce_sum(-np.log(mc_test) + utils.logsumexp(self.likelihood.log_cond_prob(self.Y, self.Y_out), 0))
        #nll = - tf.reduce_sum(tf.reduce_mean(self.likelihood.log_cond_prob(self.Y, self.layer_out), 0))
        pred, neg_ll, prob_node = self.session.run([out, nll, self.Prob_node], feed_dict={self.X:data.X, self.Y: data.Y, self.mc:mc_test})
        mean_pred = np.mean(pred, 0)
        return mean_pred, neg_ll, prob_node

    ## Return the list of TF variables that should be "free" to be optimized
    def get_vars_fixing_some(self, all_variables):
        if self.kernel_type == "identity":
            variational_parameters = [v for v in all_variables if (not v.name.startswith("q_Omega") and not v.name.startswith("log_theta"))]
            
        else:
            if (self.q_Omega_fixed_flag == True) and (self.theta_fixed_flag == True):
                variational_parameters = [v for v in all_variables if (not v.name.startswith("q_Omega") and not v.name.startswith("log_theta"))]

            if (self.q_Omega_fixed_flag == True) and (self.theta_fixed_flag == False):
                variational_parameters = [v for v in all_variables if (not v.name.startswith("q_Omega"))]

            if (self.q_Omega_fixed_flag == False) and (self.theta_fixed_flag == True):
                variational_parameters = [v for v in all_variables if (not v.name.startswith("log_theta"))]

            if (self.q_Omega_fixed_flag == False) and (self.theta_fixed_flag == False):
                variational_parameters = all_variables

        return variational_parameters

    ## Function that learns the RFSDT model with random Fourier feature approximation
    def learn(self, data, learning_rate, mc_train, batch_size, n_iterations, optimizer = None, display_step=100, test = None, mc_test=None, loss_function=None, duration = 1000000, less_prints=False):
        total_train_time = 0
        
        if optimizer is None:
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)

        ## Set all_variables to contain the complete set of TF variables to optimize
        all_variables = tf.trainable_variables()

        ## Define the optimizer
        train_step = optimizer.minimize(self.loss, var_list=all_variables)

        ## Initialize all variables
        init = tf.global_variables_initializer()
        ##init = tf.initialize_all_variables()

        ## Fix any variables that are supposed to be fixed
        train_step = optimizer.minimize(self.loss, var_list=self.get_vars_fixing_some(all_variables))

        ## Initialize TF session
        self.session.run(init)

        ## Set the folder where the logs are going to be written
        summary_writer = tf.summary.FileWriter('logs/', self.session.graph)

        if not(less_prints):
            nelbo, kl, ell, _, _ =  self.session.run(self.get_nelbo(), feed_dict={self.X: data.X, self.Y: data.Y, self.mc: mc_train})
            print("Initial kl=" + repr(kl) + "  nell=" + repr(-ell) + "  nelbo=" + repr(nelbo), end=" ")
            print("log-sigma2 =", self.session.run(self.log_theta_sigma2))

        ## Present data to RFSDT n_iterations times
        for iteration in range(n_iterations):

            ## Stop after a given budget of minutes is reached
            if (total_train_time > 1000 * 60 * duration):
                break

            ## Present one batch of data to the DGP
            start_train_time = current_milli_time()
            batch = data.next_batch(batch_size)

            monte_carlo_sample_train = mc_train
            if (current_milli_time() - start_train_time) < (1000 * 60 * duration / 2.0):
                monte_carlo_sample_train = 1

            self.session.run(train_step, feed_dict={self.X: batch[0], self.Y: batch[1], self.mc: monte_carlo_sample_train})
            total_train_time += current_milli_time() - start_train_time

            ## After reaching enough iterations with Omega fixed, unfix it
            if self.q_Omega_fixed_flag == True:
                if iteration >= self.q_Omega_fixed:
                    self.q_Omega_fixed_flag = False
                    train_step = optimizer.minimize(self.loss, var_list=self.get_vars_fixing_some(all_variables))

            if self.theta_fixed_flag == True:
                if iteration >= self.theta_fixed:
                    self.theta_fixed_flag = False
                    train_step = optimizer.minimize(self.loss, var_list=self.get_vars_fixing_some(all_variables))

            ## Display logs every "FLAGS.display_step" iterations
            if iteration % display_step == 0 or iteration == n_iterations - 1:
                start_predict_time = current_milli_time()
                
                if less_prints:
                    print("i=" + repr(iteration), end = " ")

                else:
                    nelbo, kl, ell, _, prob_node = self.session.run(self.get_nelbo(), feed_dict={self.X: data.X, self.Y: data.Y, self.mc: mc_train})
                    print("i=" + repr(iteration)  + "  kl=" + str("%.5f" % (kl)) + "  nell=" + str("%.5f" % (-ell))  + "  nelbo=" + str("%.5f" % (nelbo)), end=" ")

                    # print(" log-sigma2=", np.around(self.session.run(self.log_theta_sigma2), 5), end=" ")
                    # print(" log-lengthscale=", self.session.run(self.log_theta_lengthscale), end=" ")
                    # print(" Omega=", self.session.run(self.mean_Omega[0,:]), end=" ")
                    # print(" W=", self.session.run(self.mean_W[0][0][:]), end=" ")

                if loss_function is not None:
                    pred, nll_test, prob_node = self.predict(test, mc_test)
                    elapsed_time = total_train_time + (current_milli_time() - start_predict_time)
                    loss = loss_function.eval(test.Y, pred)
                    print(loss_function.get_name() + "=" + "%.4f" % loss, end = " ")
                    print(" nll_test=" + "%.5f" % (nll_test / len(test.Y)), end = " ")
                    print(" time=" + repr(elapsed_time), end = " ")
                    print("")
                    
                    ## save results
                    path = "Results/" + str(self.dataset)
                    if not os.path.exists(path):
                        os.mkdir(path)
                    
                    filename = str("Results/") + str(self.dataset) + "/" + str(self.fold) + "_" + str(self.h) + "_" + str(self.ard_type) + "_" + str(self.kernel_type) + "_" + str(learning_rate)
                    if iteration == 0:
                        np.savetxt(filename, np.array([nll_test / len(test.Y), loss, elapsed_time]).reshape(1,-1))
                    else:
                        with open(filename, "ab") as f:
                            np.savetxt(f, np.array([nll_test / len(test.Y), loss, elapsed_time]).reshape(1,-1))
            
            ## Explain the MNIST
            if iteration == n_iterations - 1 and self.dataset in ["MNIST", "MNIST8"]:
                for i in range(self.h):
                    for j in range(pow(2, i)):
                        filename = str("Results/") + str(self.dataset) + "/" + str(i) + "_" + str(j) + "_" + str(self.kernel_type)
                        np.savetxt(filename, np.average(prob_node[i][j], axis=0))

                
