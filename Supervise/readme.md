<h2 align="center"> 任务1：对比监督学习和自监督学习在图像分类任务上的性能表现 </h2>


### 文件结构

### 数据集&模型权重

- 数据集（无需手动下载，若本地不存在会自动下载，可在配置文件中修改数据集目录 `stream/root` ）
  - CIFAR-10
  - Tiny-ImageNet
  - CIFAR-100
- 模型权重

### 训练&测试

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