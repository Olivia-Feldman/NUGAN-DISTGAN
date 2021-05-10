This repo consists of three different gan networks that are used to compare how the regular DGAN, DIST-GAN, and NU-GAN perform on preventing mode collapse. Each gan is run at latent dimensions 1 to visualize mode collpase in the model. Then the models are run at latent dimension 10 to visualize the generation of images wihtout mode collapse. An inception score metric is used to determine the quality and diversity of images for each three gans at their latent dimensions

# Classifier

A cnn digit classifier was trained and used in both networks to predic p(y|x) for the inception score for the gan. 


# NUGAN-DISTGAN

## Lanczos Algorithm

The Lanczos algorithm is a direct algorithm devised by Cornelius Lanczos that is an adaptation of power methods to find the *m* "most useful" (tending towards extreme highest/lowest) eigenvalues and eigenvectors of an *n x n* Hermitian matrix, where *m* is often but not necessarily much smaller than *n*. [Wikipedia](https://en.wikipedia.org/wiki/Lanczos_algorithm)

Baidu-Research TensorFLow Lanczos [code](https://github.com/baidu-research/tensorflow-allreduce/blob/master/tensorflow/contrib/solvers/python/ops/lanczos.py)

Lanczos Algorithm [video](https://www.youtube.com/watch?v=0t7WJybTmFg) 1/2

Lanczos Algorithm [video](https://www.youtube.com/watch?v=WO8w5zq1Sfo) 2/2

Krylov subspaces [video](https://www.youtube.com/watch?v=ji__O4deIZo)

Lanczos Algorithm [code](https://github.com/cc-hpc-itwm/GradVis/blob/master/toolbox/hessian_functions.py)

## DIST-GAN

The Dist-GAN introduces two novel constraints that works towards preventing mode collapse. The Dist-GAN network consists of autoencoder, generator and discriminator that are trained with the novel constraints. 

A sigmoid loss function was created to returns a sigmoid loss logits probability. This is used for the discriminator loss. 

DIST-GAN [repo](https://github.com/tntrung/gan/blob/master/distgan_image/distgan_mnist.py)

## PyTorch

PyTorch [datasets](https://pytorch.org/vision/stable/datasets.html): MNIST, CIFAR10, etc.



## Mode Collapse

How to Identify and Diagnose GAN [Failure Modes](https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/)
