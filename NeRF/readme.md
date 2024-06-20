<h2 align="center"> 任务3：基于NeRF的物体重建和新视图合成 </h2>

### 文件结构

### 模型权重&视频链接

### 训练&测试

- 训练
  - 运行训练脚本`./train.sh`，即可自动进行训练，可在`vasedeck.txt`文件中修改配置参数
    ```cmd
    chmod +x ./train.sh
    ./train.sh
    ```

> [!CAUTION]
> 由于本数据集包含的图片格式为`.png`，需要对`./nerf-pytorch/load_llff.py`第111行进行如下修改，否则会报错
> ```python 
> 111 return imageio.imread(f, ignoregamma=True) # old
> 111 return imageio.imread(f)                   # new
> ```

- 测试
  - 运行测试脚本`./test.sh`，即可自动进行测试
    ```cmd
    chmod +x ./test.sh
    ./test.sh
    ``` 