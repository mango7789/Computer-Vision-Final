<h2 align="center"> 计算机视觉期末作业 </h2>

> [!WARNING]
> 开发&训练环境：Windows 11 & Ubuntu 9.4.0, Python 3.11.6   
> 计算资源：RTX 3090

> [!NOTE]
> 1. Python版本建议在**3.10及以上**，部分代码使用了3.10版本的特性（如`match`, `case`）  
> 2. 任务要求可点击展开，显示“基本要求”和“提交要求”   
> 3. 可点击**代码地址**跳转到对应的文件目录，每个目录下均有对应的`readme`文件，可根据其说明进行相应任务的训练和测试 
> 4. 项目报告均在overleaf上以 $\LaTeX$ 进行撰写，可点击对应的**报告链接**进行查看

---

### 任务1：对比监督学习和自监督学习在图像分类任务上的性能表现

<details>
<summary> 任务要求 </summary>

#### 基本要求：
- 实现任一自监督学习算法并使用该算法在自选的数据集上训练ResNet-18，随后在CIFAR-100数据集中使用Linear Classification Protocol对其性能进行评测；
- 将上述结果与在ImageNet数据集上采用监督学习训练得到的表征在相同的协议下进行对比，并比较二者相对于在CIFAR-100数据集上从零开始以监督学习方式进行训练所带来的提升；
- 尝试不同的超参数组合，探索自监督预训练数据集规模对性能的影响；

#### 提交要求：
- 提交pdf格式的实验报告，报告中除对模型、数据集和实验结果的基本介绍外，还应包含用Tensorboard可视化的训练过程中的loss曲线变化以及Linear classification过程中accuracy的变化；
- 代码提交到自己的public github repo，repo的readme中应清晰指明如何进行训练和测试，训练好的模型权重上传到百度云/google drive等网盘，实验报告内应包含实验代码所在的github repo链接及模型权重的下载地址。

</details>

#### 代码地址
- https://github.com/mango7789/Computer-Vision-Final/tree/main/Supervise

#### 报告链接
- https://cn.overleaf.com/project/665d8af812d89a8812ae678b

---

### 任务2：在CIFAR-100数据集上比较基于Transformer和CNN的图像分类模型

<details>
<summary> 任务要求 </summary>

#### 基本要求：
- 分别基于CNN和Transformer架构实现具有相近参数量的图像分类网络；
- 在CIFAR-100数据集上采用相同的训练策略对二者进行训练，其中数据增强策略中应包含CutMix；
- 尝试不同的超参数组合，尽可能提升各架构在CIFAR-100上的性能以进行合理的比较。

#### 提交要求：
- 提交pdf格式的实验报告，报告中除对模型、数据集和实验结果的介绍外，还应包含用Tensorboard可视化的训练过程中在训练集和验证集上的loss曲线和验证集上的Accuracy曲线；
- 报告中应提供详细的实验设置，如训练测试集划分、网络结构、batch size、learning rate、优化器、iteration、epoch、loss function、评价指标等。
- 代码提交到自己的public github repo，repo的readme中应清晰指明如何进行训练和测试，训练好的模型权重上传到百度云/google drive等网盘，实验报告内应包含实验代码所在的github repo链接及模型权重的下载地址。

</details>

#### 代码地址
- https://github.com/mango7789/Computer-Vision-Final/tree/main/CIFAR-100

#### 报告链接
- https://cn.overleaf.com/project/665d8aeb625559847b1489fd

---

### 任务3：基于NeRF的物体重建和新视图合成

<details>
<summary> 任务要求 </summary>

#### 基本要求：
- 选取身边的物体拍摄多角度图片/视频，并使用COLMAP估计相机参数，随后使用现成的框架进行训练；
- 基于训练好的NeRF渲染环绕物体的视频，并在预留的测试图片上评价定量结果。
  
#### 提交要求：
- 提交pdf格式的实验报告，报告中除对模型、数据和实验结果的介绍外，还应包含用Tensorboard可视化的训练过程中在训练集和测试集上的loss曲线，以及在测试集上的PSNR等指标；
- 报告中应提供详细的实验设置，如训练测试集划分、网络结构、batch size、learning rate、优化器、iteration、epoch、loss function、评价指标等。
- 代码提交到自己的public github repo，repo的readme中应清晰指明如何进行训练和测试，训练好的模型权重和渲染的视频上传到百度云/google drive等网盘，实验报告内应包含实验代码所在的github repo链接及模型权重和视频的下载地址。

</details>

#### 代码地址
- https://github.com/mango7789/Computer-Vision-Final/tree/main/NeRF