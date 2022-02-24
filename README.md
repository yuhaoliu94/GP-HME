# README

Currently the code is structured under Tensorflow 1.14.0.

## FLAGS

The code implements variational inference for Gaussian Process Decision Tree (GPDT) approximated using random Fourier features. The code accepts the following options:


* --fold                &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; Dataset fold
* --seed                &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; Seed for Tensorflow and Numpy
* --n_rff               &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; Number of random features for each layer
* --h_tree              &emsp; &emsp; &emsp; &emsp; &emsp; Height of trees
* --dataset             &emsp; &emsp; &emsp; &emsp; &ensp; Dataset name
* --mc_test             &emsp; &emsp; &emsp; &emsp; &nbsp; Number of Monte Carlo samples for predictions
* --mc_train            &emsp; &emsp; &emsp; &emsp; Number of Monte Carlo samples used to compute stochastic gradients
* --ard_type            &emsp; &emsp; &emsp; &emsp; How to treat Omega: it can be 0 for 'ISO-N', 1 for 'ISO-L', and 2 for 'NIS-N'
* --duration            &emsp; &emsp; &emsp; &emsp; &nbsp; Duration of job in minutes
* --optimizer           &emsp; &emsp; &emsp; &emsp; Optimizer: adam, adagrad, adadelta, or sgd
* --batch_size          &emsp; &emsp; &emsp; &nbsp; Batch size
* --kernel_type         &emsp; &emsp; &emsp; Kernel: RBF or arccosine kernel
* --less_prints         &emsp; &emsp; &emsp; &nbsp; Disables evaluations during the training steps
* --theta_fixed         &emsp; &emsp; &emsp; &nbsp; Number of iterations to keep theta fixed
* --n_iterations        &emsp; &emsp; &emsp; Number of iterations (batches) to train the DGP model
* --display_step        &emsp; &emsp; &emsp; Display progress every display_step iterations
* --learning_rate       &emsp; &emsp; &nbsp; &nbsp; Learning rate for optimizers
* --local_reparam       &emsp; &emsp; &nbsp; Use the local reparameterization trick
* --q_Omega_fixed       &emsp; &ensp; Number of iterations to keep posterior over Omega fixed
