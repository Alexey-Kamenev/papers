# Papers

## 2020-06

[Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)
<details>
<summary>Short abstract</summary>
In this paper, we demonstrate that the self-attention mechanism can be approximated by a low-rank matrix. We further exploit this finding to propose a new self-attention mechanism, which reduces the overall self-attention complexity from O(n^2) to O(n) in both time and space.
</details>

## 2020-09

[Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)
<details>
<summary>Short abstract</summary>
We introduce Performers, Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers with provable accuracy, but using only linear (as opposed to quadratic) space and time complexity, without relying on any priors such as sparsity or low-rankness.
</details>


## 2021-04

[LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference](https://arxiv.org/abs/2104.01136)
<details>
<summary>Short abstract</summary>
We design a family of image classification architectures that optimize the trade-off between accuracy and efficiency in a high-speed regime. Our work exploits recent findings in attention-based architectures, which are competitive on highly parallel processing hardware. We revisit principles from the extensive literature on convolutional neural networks to apply them to transformers, in particular activation maps with decreasing resolutions.
</details>


## 2021-07

[LANA: Latency Aware Network Acceleration](https://arxiv.org/abs/2107.10624)
<details>
<summary>Short abstract</summary>
We introduce latency-aware network acceleration (LANA) - an approach that builds on neural architecture search techniques and teacher-student distillation to accelerate neural networks. LANA consists of two phases: in the first phase, it trains many alternative operations for every layer of the teacher network using layer-wise feature map distillation.
</details>


## 2021-09

[Stochastic Training is Not Necessary for Generalization](https://arxiv.org/abs/2109.14119)
<details>
<summary>Short abstract</summary>
It is widely believed that the implicit regularization of stochastic gradient descent (SGD) is fundamental to the impressive generalization behavior we observe in neural networks. In this work, we demonstrate that non-stochastic full-batch training can achieve strong performance on CIFAR-10 that is on-par with SGD, using modern architectures in settings with and without data augmentation.
</details>

[A Farewell to the Bias-Variance Tradeoff? An Overview of the Theory of Overparameterized Machine Learning](https://arxiv.org/abs/2109.02355)
<details>
<summary>Short abstract</summary>
Overparameterized models are excessively complex with respect to the size of the training dataset, which results in them perfectly fitting (i.e., interpolating) the training data, which is usually noisy. Such interpolation of noisy data is traditionally associated with detrimental overfitting, and yet a wide range of interpolating models -- from simple linear models to deep neural networks -- have recently been observed to generalize extremely well on fresh test data. Indeed, the recently discovered double descent phenomenon has revealed that highly overparameterized models often improve over the best underparameterized model in test performance
</details>


## 2021-10

[ZerO Initialization: Initializing Residual Networks with only Zeros and Ones](https://openreview.net/forum?id=EYCm0AFjaSS)
<details>
<summary>Short abstract</summary>
We propose a fully deterministic initialization for training residual networks by employing skip connections and Hadamard transforms, resulting in state-of-art performance.
</details>

[The Efficiency Misnomer](https://arxiv.org/abs/2110.12894)
<details>
<summary>Short abstract</summary>
Model efficiency is a critical aspect of developing and deploying machine learning models.
</details>
<details>
<summary>Notes</summary>
</details>

[NViT: Vision Transformer Compression and Parameter Redistribution](https://arxiv.org/abs/2110.04869)
<details>
<summary>Short abstract</summary>
We apply global, structural pruning with latency-aware regularization on all parameters of the Vision Transformer (ViT) model for latency reduction. Furthermore, we analyze the pruned architectures and find interesting regularities in the final weight structure.
</details>


## 2021-12

[AdaViT: Adaptive Tokens for Efficient Vision Transformer](https://arxiv.org/abs/2112.07658)
<details>
<summary>Short abstract</summary>
We introduce AdaViT, a method that adaptively adjusts the inference cost of vision transformer (ViT) for images of different complexity. AdaViT achieves this by automatically reducing the number of tokens in vision transformers that are processed in the network as inference proceeds.
</details>
