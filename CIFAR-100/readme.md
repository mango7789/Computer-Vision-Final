<h2 style="text-align: center;"> 任务2：在CIFAR-100数据集上比较基于Transformer和CNN的图像分类模型 </h2>

### 文件结构


### 数据集&模型权重

- 数据集
  - 无需手动下载，训练时若本地不存在会自动进行下载，需要指定`data_root`数据集根目录
- 模型权重

### 训练&测试

- 训练(提供多种方式)
  - 训练一个配置
    - 修改`config.yaml`配置文件中的参数，运行`train.py`文件
    - 或者直接在终端中运行，可指定参数
      ```bash
      python .\train.py --epochs 20 --fc_lr 0.01 --save
      ```
    > [!NOTE]
    > 部分参数如`seed`, `momentum`, `gamma`, `step_size` 和 `weight_decay` 未在`train.py`中指定，需要自行在配置文件中修改  
  - 批量训练
    - 运行`train.sh`脚本，可在其中修改对应的参数
    - 运行`train.ipynb`，可在对应的代码块中修改参数
- 测试