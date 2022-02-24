import abc

class Likelihood:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_cond_prob(self, output, latent_val):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def predict(self, latent_val):
        raise NotImplementedError("Subclass should implement this.")

