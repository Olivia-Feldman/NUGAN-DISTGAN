This repo consists of three different GANs that are used to compare how a regular GAN, DIST-GAN, and NU-GAN perform when it comes to preventing mode collapse. Each GAN is run with 1 latent dimension to visualize mode collapse in the model. Then, the models are run with 10 latent dimensions to visualize the generation of images wihtout mode collapse. An inception score metric is calculated at each epoch to determine the quality and diversity of images for each GAN at the specified latent dimensions.

# Classifier

A CNN digit classifier was trained and implemented in all networks to find p(y|x) for the inception scores. 


# NUGAN-DISTGAN

## Lanczos Algorithm

The Lanczos algorithm is a direct algorithm devised by Cornelius Lanczos that is an adaptation of power methods to find the *m* "most useful" (tending towards extreme highest/lowest) eigenvalues and eigenvectors of an *n x n* Hermitian matrix, where *m* is often but not necessarily much smaller than *n*. [Wikipedia](https://en.wikipedia.org/wiki/Lanczos_algorithm)

Baidu-Research TensorFLow Lanczos [code](https://github.com/baidu-research/tensorflow-allreduce/blob/master/tensorflow/contrib/solvers/python/ops/lanczos.py)

Lanczos Algorithm [video](https://www.youtube.com/watch?v=0t7WJybTmFg) 1/2

Lanczos Algorithm [video](https://www.youtube.com/watch?v=WO8w5zq1Sfo) 2/2

Krylov subspaces [video](https://www.youtube.com/watch?v=ji__O4deIZo)

Lanczos Algorithm [code](https://github.com/cc-hpc-itwm/GradVis/blob/master/toolbox/hessian_functions.py)

## DIST-GAN

The Dist-GAN introduces two novel constraints that work towards preventing mode collapse. The Dist-GAN network consists of an autoencoder, a generator, and a discriminator that are trained with the novel constraints. 

A sigmoid loss function was created to return a sigmoid loss logits probability. This is used for the discriminator loss. 

DIST-GAN [repo](https://github.com/tntrung/gan/blob/master/distgan_image/distgan_mnist.py)

## PyTorch

PyTorch [datasets](https://pytorch.org/vision/stable/datasets.html): MNIST, CIFAR10, etc.



## Mode Collapse

How to Identify and Diagnose GAN [Failure Modes](https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/)
