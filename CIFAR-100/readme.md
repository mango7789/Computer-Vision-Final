<h2 align="center"> 任务2：在CIFAR-100数据集上比较基于Transformer和CNN的图像分类模型 </h2>

### 文件结构


### 数据集&模型权重

- 数据集
  - 无需手动下载，训练时若本地不存在会自动进行下载，可指定数据集根目录 `data_root` 
- 模型权重

### 训练&测试

- 训练（提供多种方式）
  - 训练单个配置
    - 修改 `config.yaml` 配置文件中的参数，运行 `train.py` 文件
    - 或者直接在终端中运行，可指定参数
      ```bash
      python ./train.py --epochs 20 --fc_lr 0.01 --save
      ```
> [!NOTE]
> 部分参数如`seed`, `momentum`, `gamma`, `step_size` 和 `weight_decay` 未在`train.py`中指定，需要自行在配置文件中修改  
  - 批量训练
    - 运行`train.sh`脚本，可在文件对应的位置修改参数
      ```bash
      chmod +x train.sh
      ./train.sh
      ```
    - 运行`train.ipynb`，可在对应的代码块中修改参数
- 测试
  - 在终端中运行以下命令，将会输出模型在测试集上的准确率
    ```bash
    python ./test.py --path CNN-CIFAR100.pth
    python ./test.py --path ViT-CIFAR100.pth
    ``` 