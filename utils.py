import numpy as np
from scipy.stats import t, norm
import tensorflow as tf

## Draw prior hyper-parameters
def get_random(size):
    stack = pow(10, np.linspace(-4,6,11))
    return np.random.choice(stack, size)

## Log-density of a univariate Gaussian distribution
def log_norm_pdf(x, loc=0.0, scale=0.0):
    return norm.pdf(x, loc=loc, scale=scale)

## Log-density of a univariate t distribution
def log_t_pdf(x, loc=0.0, scale=0.0, df=3):
    return t.pdf(x, df=df, loc=loc, scale=scale)

## Draw an array of multivariate normal
def get_mvn_samples(mean, cov, shape):
    return np.random.multivariate_normal(mean=mean, cov=cov, size=shape)

## Draw an array of standard normal
def get_normal_samples(shape):
    return np.minimum(np.random.normal(size=shape), 10)

## Draw an array of standard student's t
def get_t_samples(shape, nu):
    return np.minimum(np.random.standard_t(df=nu, size=shape), 10)

## Calculate the phi^\top \Sigma \phi
def diag(Phi, Sigma):
    tmp = np.dot(Phi, Sigma)
    var = [np.dot(tmp[mc,:], Phi[mc,:]) for mc in range(Phi.shape[0])]
    return np.array(var)

## Normalize log weights
def normalize_weight(log_weight):
    log_weight -= max(log_weight)
    weight = np.exp(log_weight)
    weight /= sum(weight)
    return weight

## Log-sum operation
def logsumexp(vals, dim=None):
    m = np.max(vals, dim)
    if dim is None:
        return m + np.log(np.sum(np.exp(vals - m), dim))
    else:
        return m + np.log(np.sum(np.exp(vals - np.expand_dims(m, dim)), dim))


## Get flags from the command line
def get_flags():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('batch_size', 50, 'Batch size.  ')
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('n_iterations', 2000, 'Number of iterations (batches) to feed to the DGP')
    flags.DEFINE_integer('display_step', 100, 'Display progress every FLAGS.display_step iterations')
    flags.DEFINE_integer('mc_train', 30, 'Number of Monte Carlo samples used to compute stochastic gradients')
    flags.DEFINE_integer('mc_test', 30, 'Number of Monte Carlo samples for predictions')
    flags.DEFINE_integer('n_rff', 10, 'Number of random features for each layer')
    flags.DEFINE_integer('df', 1, 'Number of GPs per hidden layer')
    flags.DEFINE_integer('nl', 1, 'Number of layers')
    flags.DEFINE_string('optimizer', "adagrad", 'Optimizer')
    flags.DEFINE_string('kernel_type', "RBF", 'arccosine')
    flags.DEFINE_integer('kernel_arccosine_degree', 1, 'Degree parameter of arc-cosine kernel')
    flags.DEFINE_boolean('is_ard', True, 'Using ARD kernel or isotropic')
    flags.DEFINE_boolean('local_reparam', True, 'Using the local reparameterization trick')
    flags.DEFINE_boolean('feed_forward', False, 'Feed original inputs to each layer')
    flags.DEFINE_boolean('VI', True, 'Using variational inference or not')
    flags.DEFINE_integer('q_Omega_fixed', 0, 'Number of iterations to keep posterior over Omega fixed')
    flags.DEFINE_integer('theta_fixed', 0, 'Number of iterations to keep theta fixed')
    flags.DEFINE_string('learn_Omega', 'var_fixed', 'How to treat Omega - fixed (from the prior), optimized, or learned variationally')
    flags.DEFINE_integer('duration', 100000, 'Duration of job in minutes')

    # Flags for online learning
    flags.DEFINE_integer('N_iterations', 0, 'Number of iterations (samples) to feed to the DGP, 0: Only use samples once.')
    flags.DEFINE_integer('MC_train', 100, 'Number of Monte Carlo samples used to online training')
    flags.DEFINE_integer('MC_test', 100, 'Number of Monte Carlo samples used to online prediction')

    # Flags for use in cluster experiments
    tf.app.flags.DEFINE_string("dataset", "", "Dataset name")
    tf.app.flags.DEFINE_string("fold", "1", "Dataset fold")
    tf.app.flags.DEFINE_integer("seed", 0, "Seed for random tf and np operations")
    tf.app.flags.DEFINE_boolean("less_prints", False, "Disables evaluations involving the complete dataset without batching")
    
    return FLAGS

## Define the right optimizer for a given flag from command line
def get_optimizer(opt_name, learning_rate):
    switcher = {
        "adagrad": tf.train.AdagradOptimizer(learning_rate),
        "sgd": tf.train.GradientDescentOptimizer(learning_rate),
        "adam": tf.train.AdamOptimizer(learning_rate),
        "adadelta": tf.train.AdadeltaOptimizer(learning_rate)
    }
    return switcher.get(opt_name)
