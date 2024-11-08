# Tight and Efficient Upper Bound on Spectral Norm of Convolutional Layers

This repository is the official implementation of our ECCV 2024 paper "Tight and Efficient Upper Bound on Spectral Norm of Convolutional Layers" by Ekaterina Grishina, Mikhail Gorbunov and Maxim Rakhuba.

* `bounds.py` contains code for different bounds on the spectral norm of convolution. `compute_tensor_norm_einsum` and `compute_tensor_norm` are the implementations of our proposed method. For the other bounds, we used code from the official repositories of the corresponding papers.
* `pretrained_net_norms.py` contains example of the spectral norm computation for the pretrained ResNet18 and VGG.