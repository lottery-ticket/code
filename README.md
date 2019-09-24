# Comparing Rewinding and Fine-tuning in Neural Network Pruning

This is an anonymized public release of the code for *Comparing Rewinding and Fine-tuning in Neural Network Pruning*.
This code was used to generate all of the data and plots in the paper.

The ResNet-20 implementation (using GPUs) can be found in `gpu-src/official/resnet`, and is based off of [the implementation provided alongside Tensorflow](https://github.com/tensorflow/models/tree/v1.13.0/official/resnet).
The VGG-16 implementation (using GPUs) can be found in `gpu-src/official/vgg`, and is a fork of the ResNet-20 implementation, modified to reflect a VGG-16 with batch norm and a with single fully connected layer at the end.
The ResNet-50 implementation (using TPUs) can be found in `tpu-src/models/official/resnet` and is based off of [the implementation provided alongside Tensorflow](https://github.com/tensorflow/tpu/tree/98497e0b/models/official/resnet).

The rewinding and pruning code can be found in `lottery/lottery/lottery.py` and `lottery/lottery/prune_functions.py`.

## Running the code

To set up the environment, a Dockerfile is provided in `docker/_docker` which installs all dependencies.
The Dockerfile expects the following environment variables:
- ROOT_INSTALL_FILE=`docker/root_setup.sh`
- PROJECT_NAME=lth
- PROJECT_NAME_UPPERCASE=LTH
- USER_INSTALL_FILE=`docker/user_setup.sh`

Once the Docker environment is built and running, run `docker/posthoc_setup.sh`

To train a network, `cd` to `gpu-src` or `tpu-src`, and train the model as specified in the original [GPU](https://github.com/tensorflow/models/tree/v1.13.0/official/resnet) or [TPU](https://github.com/tensorflow/tpu/tree/98497e0b/models/official/resnet) codebase, with two added command line flags:
- `--lottery_results_dir ${RESULTS_DIR}`, which specifies the directory to save rewinding checkpoints to, and
- `--lottery_checkpoint_iters 4701,11259,...`, which is a comma-separated list of iterations to checkpoint so that they can be rewound to later.

Once a network is trained, it can be pruned and re-trained by running the network training code again, with a different `model_dir` (as defined by the original codebases) and with the following additional parameters:
- `--lottery_prune_at "${RESULTS_DIR}/checkpoint_iter_final"`, which says to prune the network using the weights from the end of training
- `--lottery_reset_to "${RESULTS_DIR}/checkpoint_iter_4701"`, which says to rewind the weights to their values at iteration 4701.

To fine-tune rather than rewind, simply set `--lottery_reset_to "${RESULTS_DIR}/checkpoint_iter_final"`, and `--lottery_force_learning_rate 0.001` to force the learning rate to the desired value for fine-tuning.

Both `--lottery_prune_at` and `--lottery_reset_to` accept `reinitialize` as an argument to reinitialize the weights before pruning or re-training, which allows for generation of random pruning masks, or for running the reinitialization baselines.

To view the final accuracy reuslts, inspect the `tfevent` summary file produced by tensorflow.
