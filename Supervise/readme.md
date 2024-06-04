<h2 style="text-align: center;"> 任务1：对比监督学习和自监督学习在图像分类任务上的性能表现 </h2>


### Linear Classification Protocol (LCP) is a common evaluation method for self-supervised learning (SSL) algorithms. The general process involves the following steps:

- **Pretrain the Model using SSL**: Train a ResNet-18 model on a chosen dataset using a self-supervised learning algorithm.
- **Freeze the Backbone**: Once pretrained, freeze the weights of the ResNet-18 model.
- **Train a Linear Classifier**: On top of the frozen backbone, add a linear layer (fully connected layer) and train it using the labeled CIFAR-100 dataset.
- **Evaluate Performance**: Evaluate the performance of the linear classifier on the CIFAR-100 test set.