This repo consists of three different GANs that are used to compare how a regular GAN, DIST-GAN, and NU-GAN perform when it comes to preventing mode collapse. Each GAN is run with 1 latent dimension to visualize mode collapse in the model. Then, the models are run with 10 latent dimensions to visualize the generation of images wihtout mode collapse. An inception score metric is calculated at each epoch to determine the quality and diversity of images for each GAN at the specified latent dimensions.

## Classifier

A CNN digit classifier was trained and implemented in all networks to find p(y|x) for the inception scores. 

## Inception Score
The Inception Score, or IS for short, is an objective metric for evaluating the quality of generated images, specifically synthetic images output by generative adversarial network models. It was developed to remove the subjective human evaluation of images. The score seeks to capture two properties of a collection of generated images:
1) **Image Quality:** Do images look like a specific object?
2) **Image Diversity:** Is a wide range of objects generated?

[More on Inception Scores](https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/)

## DIST-GAN

Dist-GAN introduces the latent distance constraint and discriminator score constraint that work towards preventing mode collapse. The Dist-GAN network consists of an autoencoder, a generator, and a discriminator that are trained with these two novel constraints. The training contains three phases, 1) Training the encoder and generator with the latent distance constraint, 2) training the discriminator with the discriminator score constraint, and 3) Training the generator a second time. 

A sigmoid loss function was developed to return a sigmoid loss logits probability. This is used to calculate the discriminator loss and takes in the logits probability and class labels to determine the sigmoid loss. 

The latent distance constraint is used to regularize the auto-encoder and enforce compatibility between the latent samples and data samples. A distance distribution metric is for the latent and data sample distributions.

The discriminator score constraint is used to calculate the distance of distributions between the generator and discriminator, along with a gradient penalty to determine the total D_loss from the discriminator. 


[DIST-GAN repo](https://github.com/tntrung/gan/blob/master/distgan_image/distgan_mnist.py)

## NuGAN+DIST-GAN

The combined NuGAN+DIST-GAN model has the same architecture as DIST-GAN but also implements a modified nudged-Adam optimizer. This modified optimizer calculates the hessian of the loss function of either the generator or discriminator. The top-k eigenvalues of this hessian matrix can then be calculated. If any of these top-k eigenvalues are above a certain threshold, then that is a signal for possible mode collapse. In our NuGAN+DISTGAN.ipynb file, we used a threshold of 500 to determine if an eigenvalue was signaling mode collapse. Once you know which eigenvalues pass that threshold, you can remove the gradient information for the weight parameter that corresponds to that eigenvalue.

The [NuGAN paper](https://arxiv.org/pdf/2012.09673.pdf) that we utilized to implement this nudged-Adam also needed an efficient way to compute the eigenvalues of the Hessian because most typical neural networks have millions of parameters. Trying to calculate the eigenvalues of a Hessian by normal means would be infeasible. To circumvent this issue, the researchers used the Lanczos algorithm, which allowed them to compute the eigenvalues of the Hessian without having to calculate and store the Hessian itself.

## Lanczos Algorithm

The Lanczos algorithm is a direct algorithm devised by Cornelius Lanczos that is an adaptation of power methods to find the *m* "most useful" (tending towards extreme highest/lowest) eigenvalues and eigenvectors of an *n x n* Hermitian matrix, where *m* is often but not necessarily much smaller than *n*. [Wikipedia](https://en.wikipedia.org/wiki/Lanczos_algorithm)

Baidu-Research TensorFLow Lanczos [code](https://github.com/baidu-research/tensorflow-allreduce/blob/master/tensorflow/contrib/solvers/python/ops/lanczos.py)

Lanczos Algorithm [video](https://www.youtube.com/watch?v=0t7WJybTmFg) 1/2

Lanczos Algorithm [video](https://www.youtube.com/watch?v=WO8w5zq1Sfo) 2/2

Krylov subspaces [video](https://www.youtube.com/watch?v=ji__O4deIZo)

Lanczos Algorithm [code](https://github.com/cc-hpc-itwm/GradVis/blob/master/toolbox/hessian_functions.py)

## PyTorch

[PyTorch datasets](https://pytorch.org/vision/stable/datasets.html): MNIST, CIFAR10, etc.


## Mode Collapse

How to Identify and Diagnose GAN [Failure Modes](https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/)
