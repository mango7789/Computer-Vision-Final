<h2 align="center"> 任务1：对比监督学习和自监督学习在图像分类任务上的性能表现 </h2>

<h2> 目录 </h2>

- [文件结构](#文件结构)
- [数据集\&模型权重](#数据集模型权重)
- [训练\&测试](#训练测试)

### 文件结构

```bash
Supervise
├──data             # 数据集目录，运行训练代码后自动生成
    ├──cifar-10-batchs-py/      # CIFAR-10
    ├──tiny-imagenet-200/       # tiny-imagenet
    └──cifar-100-python/        # CIFAR-100
├──model            # 模型权重文件夹
    ├──byol.pth                     # BYOL训练resnet模型权重
    ├──resnet_with_pretrain.pth     # 预训练resnet微调后权重
    ├──resnet_no_pretrain.pth       # 随机初始化resnet训练后权重
    ├──self_supervise.pth           # 自监督线性分类器权重
    ├──supervise_with_pretrain.pth  # 预训练监督线性分类器权重
    └──supervise_no_pretrain.pth    # 随机初始化线性分类器权重
├──logs             # 训练日志
    ├──BYOL/...                 # BYOL训练日志，包含train loss
    ├──ResNet-18/...            # resnet18训练日志，包含train loss
    ├──Linear-Classifier/...    # 线性分类器训练日志，包含loss和acc
    ├──tensorboard/...          # 由训练日志转换成的tensorboard文件
    ├──log2tensorboard.py       # 将训练日志转换成tensorboard的脚本
    ├──byol-accuracy.txt                  # BYOL自监督学习在测试集上准确率
    ├──resnet_with_pretrain-accuracy      # 预训练resnet监督学习在测试集上准确率
    └──resnet_no_pretrain-accuracy.txt    # 随机初始化resnet监督学习在测试集上准确率
├──img              # 数据集、模型架构和tensorboard截图
├──config.yaml      # 参数配置文件
├──utils.py         # 辅助函数，包括导入数据集，获取模型，处理日志
├──byol.py          # BYOL自监督学习模型的实现
├──solver.py        # 求解器，包含训练和测试的主函数
├──train.py         # 训练的parser，可指定不同超参数
├──train.sh         # 批量化训练的shell脚本
├──test.py          # 测试的parser，可指定模型路径
└──readme.md      
```

> [!TIP]
> BYOL日志文件的名称含义为`{epochs}--{lr}--{hidden_dim}--{output_dim}--{dataset}.log`，分隔符为`--`

### 数据集&模型权重

- 数据集（无需手动下载，训练时若本地不存在会自动下载，可在配置文件中修改数据集目录 `stream/root` ）
  - CIFAR-10
  - Tiny-ImageNet
  - CIFAR-100
- 模型权重
  - 新建`model`文件夹，手动下载[模型权重](https://drive.google.com/drive/folders/16h4CnCKFLFrOgW7ID3xhPNj5JlL64Ra0?usp=sharing)，将其放入`model`文件夹下
  - 或者在终端中输入以下命令，自动进行下载
    ```bash
    pip install gdown
    mkdir -p ./model
    gdown --folder https://drive.google.com/drive/folders/16h4CnCKFLFrOgW7ID3xhPNj5JlL64Ra0 -O ./model
    ``` 

### 训练&测试

> 注意：默认工作目录为`./Supervise`

- 训练
  - 配置
    - 可在`config.yaml`文件中修改对应的默认参数
  - 运行
    - 若要对单个网络进行训练，可在终端中运行以下命令
      ```bash
      # train the byol model
      python train.py byol --epochs 10 --lr 0.001 -save
      # train the ResNet-18 model
      python train.py resnet --epochs 15 --lr 0.005 --seed 42
      # train the linear classifier
      python train.py linear --epochs 150 --model ./model/byol.pth --type 'self_supervise'
      # for more info about the parser, run help
      python train.py [byol, resent, linear] --help
      ``` 
    - 若要对全部网络进行训练，只需在bash终端中输入以下命令，可在shell脚本中修改对应参数
      ```bash
      chmod +x train.sh
      ./train.sh
      ``` 
- 测试
  ```bash
  # test the byol
  python test.py --encoder ./model/byol.pth --classifier ./model/self_supervise.pth
  # test the supervise with pretrain
  python test.py --encoder ./model/resnet_with_pretrain.pth --classifier ./model/supervise_with_pretrain.pth
  # test the supervise without pretrain
  python test.py --encoder ./model/resnet_no_pretrain.pth --classifier ./model/supervise_no_pretrain.pth
  ```
- 可视化训练过程
  - 在终端中输入以下命令，点击 [http://localhost:6006/]() 打开tensorboard进行查看
    ```bash
    tensorboard --logdir ./logs/tensorboard
    ``` 