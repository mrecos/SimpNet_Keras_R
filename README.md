# SimpNet in Keras for R
This is a repository for translating SimpNet in an R flavored Keras implementation.
SimpNet is a deep convolutional neural network architecture reported on in:

    Towards Principled Design of Deep Convolutional Networks: Introducing SimpNet
    Seyyed Hossein Hasanpour, Mohammad Rouhani, Mohsen Fayyaz, Mohammad Sabokrou and Ehsan Adeli
    
[Hasanpour et al. 2018 on arXiv](https://arxiv.org/abs/1802.06205)

This is a link to the original [SimpNet Github repository](https://github.com/Coderx7/SimpNet)
From the author's abstract:

   We empirically show that SimpNet provides a good trade-off between the computation/memory efficiency and the accuracy solely based on these primitive but crucial principles. SimpNet outperforms the deeper and more complex architectures such as VGGNet, ResNet, WideResidualNet \etc, on several well-known benchmarks, while having 2 to 25 times fewer number of parameters and operations. We obtain state-of-the-art results (in terms of a balance between the accuracy and the number of involved parameters) on standard data sets, such as CIFAR10, CIFAR100, MNIST and SVHN.

### Schematic SimpNet architecture (Hasanpour et al. 2018)

![]("https://github.com/Coderx7/SimpNet/blob/master/SimpNetV2/images/Arch2_01.jpg")
   
## Keras Version
This repo is an attempt to translate the SimpNet architecture into the Keras API via R and the [Rstudio flavor of Keras](https://keras.rstudio.com/).

So far I have translated:

    * SimpNetV2 for MNIST
        * MNIST_SimpleNet_GP_13L_drpall_5Mil_66_maxdrp
    * SimpNetV2 for CIFAR10
        * CIFAR10_SimpleNet_GP_13L_drpall_8Mil_66_DRP_After_Pooling
        
## Results of Keras version
For the original results of SimpNet in its native Caffe implementation, see the [article](https://arxiv.org/abs/1802.06205) and [GH repo](https://github.com/Coderx7/SimpNet). The results below are my translation and may not be fully reflective of the native implementation. I will keep trying to match the published results.

### MNIST version
| Epochs:        | 10         |
|----------------|------------|
| Test Loss:     | 0.01990014 |
| Test Accuracy: | 0.9949     |
|                |            |


### CIFAR10 version
| Epochs:        | 50         |
|----------------|------------|
| Test Loss:     | 0.5701714  |
| Test Accuracy: | 0.887      |
|                |            |

### Training history for CIFAR10
![]("images/CIFAR10_50_epochs.png")

