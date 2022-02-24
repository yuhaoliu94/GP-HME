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
        return latent_val
    
    def get_task(self):
        return "Regression"
