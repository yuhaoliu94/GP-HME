import tensorflow as tf

from . import likelihood
import utils


class Gaussian(likelihood.Likelihood):
    def __init__(self, log_var=-2.0):
        self.log_var = tf.Variable(log_var, name="log_theta")

    def log_cond_prob(self, output, latent_val):
        return utils.log_norm_pdf(output, latent_val, self.log_var)

    def get_params(self):
        return self.log_var

    def predict(self, latent_val):
        # std = tf.exp(self.log_var / 2.0)
        return latent_val# + std * tf.random_normal([1, tf.shape(latent_val)[1], 1])
    
    def get_task(self):
        return "Regression"
