import tensorflow as tf
from . import likelihood
import utils

class Softmax(likelihood.Likelihood):
    """
    Implements softmax likelihood for multi-class classification
    """
#    def __init__(self):

    def log_cond_prob(self, output, latent_val):
        return tf.reduce_sum(output * latent_val, 2) - utils.logsumexp(latent_val, 2)

    def predict(self, latent_val):
        """
        return the probabilty for all the samples, datapoints and classes
        :param latent_val:
        :return:
        """
        logprob = latent_val - tf.expand_dims(utils.logsumexp(latent_val, 2), 2)
        return tf.exp(logprob)

    def get_params(self):
        return None
    
    def get_task(self):
        return "Classification"
