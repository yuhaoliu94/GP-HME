# README

Currently the code is structured under Tensorflow 1.14.0. If GPUs are not available, please directly remove the GPU setting when initialize the session.

## FLAGS

The code implements variational inference for Gaussian Process Decision Tree (GPDT) approximated using random Fourier features. The code accepts the following options:


* --fold                &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; Fold of dataset
* --seed                &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; Seed for Tensorflow and Numpy
* --n_rff               &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; Number of random features
* --h_tree              &emsp; &emsp; &emsp; &emsp; &emsp; Height of trees
* --dataset             &emsp; &emsp; &emsp; &emsp; &ensp; Name of dataset
* --mc_test             &emsp; &emsp; &emsp; &emsp; &nbsp; Number of Monte Carlo samples for predictions
* --mc_train            &emsp; &emsp; &emsp; &emsp; Number of Monte Carlo samples for training
* --ard_type            &emsp; &emsp; &emsp; &emsp; How to treat Omega: it can be 0 for 'ISO-N', 1 for 'ISO-L', and 2 for 'NIS-N'
* --duration            &emsp; &emsp; &emsp; &emsp; &nbsp; Duration in minutes
* --optimizer           &emsp; &emsp; &emsp; &emsp; Optimizer: adam, adagrad, adadelta, or sgd
* --batch_size          &emsp; &emsp; &emsp; &nbsp; Batch size
* --kernel_type         &emsp; &emsp; &emsp; Kernel: RBF, arccosine, or identity
* --less_prints         &emsp; &emsp; &emsp; &nbsp; Disables evaluations during the training steps
* --theta_fixed         &emsp; &emsp; &emsp; &nbsp; Number of iterations to keep theta fixed at the beginning
* --n_iterations        &emsp; &emsp; &emsp; Number of iterations (batches) to train the DGP model
* --display_step        &emsp; &emsp; &emsp; Display progress every display_step iterations
* --learning_rate       &emsp; &emsp; &nbsp; &nbsp; Learning rate for optimizers
* --local_reparam       &emsp; &emsp; &nbsp; Use the local reparameterization trick
* --q_Omega_fixed       &emsp; &ensp; Number of iterations to keep posterior of Omega fixed at the beginning

## EXAMPLES

Here are two examples to run the GPDT model on regression and classification tasks:

### REGRESSION
```
python experiments/sdt_rff_regression_gpu.py --seed=12345 --dataset=ABA --fold=1 --q_Omega_fixed=1000 \
--theta_fixed=4000 --ard_type=0 --optimizer=adam --h_tree=2 --learning_rate=0.001 --n_rff=50 \
--batch_size=200 --mc_train=100 --mc_test=100 --n_iterations=100000 --display_step=250 --duration=60 \
--kernel_type=RBF
```

### CLASSIFICATION
```
python experiments/sdt_rff_classification_gpu.py --seed=12345 --dataset=OPT --fold=1 --q_Omega_fixed=1000 \
--theta_fixed=4000 --ard_type=0 --optimizer=adam --h_tree=2 --learning_rate=0.001 --n_rff=100 \
--batch_size=200 --mc_train=100 --mc_test=100 --n_iterations=100000 --display_step=250 --duration=60 \
--kernel_type=arccosine
```
