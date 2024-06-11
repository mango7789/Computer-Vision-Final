<h2 align="center"> 任务2：在CIFAR-100数据集上比较基于Transformer和CNN的图像分类模型 </h2>

### 文件结构

```bash
CIFAR-100
├──data             # 数据集目录，运行训练代码后自动生成
    ├──cifar-100-python.tar.gz  # CIFAR-100压缩包
    └──cifar-100-python         # CIFAR-100数据集
├──model            # 模型权重，需要自建并放入模型权重文件
    ├──CNN-CIFAR100.pth         # CNN模型权重
    └──ViT-CIFAR100.pth         # ViT模型权重
├──logs             # 训练日志
    ├──CNN/...                  # CNN训练日志，包含loss和acc
    ├──ViT/...                  # ViT训练日志，包含loss和acc
    ├──tensorboard/...          # 由训练日志转换成的tensorboard文件
    ├──log2tensorboard.py       # 将训练日志转换成tensorboard的脚本
    ├──grid-search-accuracy.txt # Grid-Search不同超参数组合的acc
    ├──cnn-accuracy.txt         # 训练好的CNN模型在测试集上准确率
    └──vit-accuracy.txt         # 训练好的ViT模型在测试集上准确率
├──img              # tensorboard截图
    ├──accuracy/...             # 训练过程acc的截图
    ├──loss/...                 # 训练过程loss的截图
    └──model/...                # 模型架构图
├──config.yaml      # 参数配置文件
├──utils.py         # 辅助函数，包括导入数据集，获取模型，处理日志
├──solver.py        # 求解器，包含训练和测试的主函数
├──train.py         # 训练的parser，可指定不同超参数
├──train.sh         # 批量化训练的shell脚本
├──test.py          # 测试的parser，可指定模型路径
└──readme.md      
```

> [!TIP]
> 日志文件的名称含义为`{epochs}--{ft_lr}--{fc_lr}--{batch_size}.log`，分隔符为`--`


### 数据集&模型权重

- 数据集
  - 无需手动下载，训练时若本地不存在会自动进行下载，可指定数据集根目录 `root` 
- 模型权重
  - 新建`model`文件夹，手动下载[模型权重](https://drive.google.com/drive/folders/1pV74DSM_MMEqIT9KZygSfciS4wUiW370?usp=drive_link)，将其放入`model`文件夹下
  - 或者在终端中输入以下命令，自动进行下载
    ```bash
    pip install gdown
    mkdir -p ./model
    gdown --folder https://drive.google.com/drive/folders/1pV74DSM_MMEqIT9KZygSfciS4wUiW370 -O ./model
    ``` 

### 训练&测试

> 注意：默认工作目录为`./CIFAR-100`

- 训练
  - 参数
    - 在 `config.yaml` 配置文件中进行修改
  - 训练单个配置
    - 直接使用命令行工具运行，可指定参数，未指定的参数采用配置文件中的默认值
      ```bash
      python ./train.py --epochs 20 --fc_lr 0.01 --save
      ```
  - 批量训练
    - 在终端中输入以下命令，可在配置文件中修改或 `train.sh` 脚本中指定参数
      ```bash
      chmod +x train.sh
      ./train.sh
      ```
> [!NOTE]
> 部分参数如`seed`, `momentum`, `gamma`, `step_size` 和 `weight_decay` 未在`train.py`中指定，需要自行在配置文件中修改  
- 测试
  - 在终端中运行以下命令，将会输出模型在测试集上的准确率，测试结果会输出到终端，且会保存在`./logs`文件夹下，名称为`[cnn, vit]-accuracy.txt`
    ```bash
    python ./test.py --path ./model/CNN-CIFAR100.pth
    python ./test.py --path ./model/ViT-CIFAR100.pth
    ``` 
- 可视化训练过程
  - 在终端中输入以下命令，点击 [http://localhost:6006/]() 打开tensorboard进行查看
    ```bash
    tensorboard --logdir ./logs/tensorboard
    ``` 