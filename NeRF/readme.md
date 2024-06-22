<h2 align="center"> 任务3：基于NeRF的物体重建和新视图合成 </h2>

### 目录

- [目录](#目录)
- [文件结构](#文件结构)
- [训练图片\&模型权重\&视频](#训练图片模型权重视频)
- [训练\&测试](#训练测试)


### 文件结构

```bash
NeRF
├──data/llff/
    ├──hhsw             # 虎虎生威，老虎模型
    └──vasedeck         # 花瓶模型
├──LLFF/          # LLFF，用于生成姿态文件
├──nerf-pytorch/  # nerf框架
├──logs           # 训练日志及结果
    ├──hhsw/            # 老虎模型训练生成的文件，包含模型参数，测试集图片和视频
    ├──vasedeck/        # 花瓶模型训练生成的文件，包含模型参数，测试集图片和视频
    ├──tensorboard/     # tensorboard文件
    ├──hhsw.txt         # 老虎模型训练日志
    ├──vasedeck.txt     # 花瓶模型训练日志
    └──log2tensorboard.py
├──img
    ├──object/          # 模型overview
    ├──colmap/          # colmap重建截图
    ├──train/           # 训练loss与psnr
    ├──val_images/      # 验证集图片
    └──pipeline.jpg
├──config.txt     # 配置文件
├──train.sh       # 训练脚本
├──test.sh        # 测试脚本
└──readme.md
```

### 训练图片&模型权重&视频

- 下载方式
  - 训练图片
    ```bash
    pip install gdown
    mkdir -p ./data
    gdown --folder https://drive.google.com/drive/folders/1w9Uah5TvxrJ7ESfgScv3z4nMmtFuyKUq -O ./data
    ```
  - 模型权重&视频
    ```bash
    gdown --folder https://drive.google.com/drive/folders/1I4xNuPrsoqD93XG9onomn3-x6P8bPYoE -O ./logs
    ```

- 模型权重
  - 老虎模型：`./logs/hhsw/`
  - 花瓶模型：`./logs/vasedeck/`
- 视频地址
  - 老虎模型：`./logs/hhsw/hhsw_spiral_200000_rgb.mp4`
  - 花瓶模型：`./logs/vasedeck/vasedeck_spiral_200000_rgb.mp4`

    https://github.com/mango7789/Computer-Vision-Final/assets/115400861/dba4bbc8-a233-4960-957d-9f2566b38b52


### 训练&测试
> 注意：默认工作目录为`./NeRF`
- ~~生成姿态文件（可跳过）~~
  - 使用`LLFF`中的`imgs2poses.py`进行生成
    ```cmd
    python ./LLFF/imgs2poses.py ./data/llff/hhsw
    python ./LLFF/imgs2poses.py ./data/llff/vasedeck
    ``` 

- 训练
  - 运行训练脚本`./train.sh`，即可自动进行训练，可在`config.txt`文件中修改配置参数
    ```cmd
    chmod +x ./train.sh
    ./train.sh
    ```
- 测试
  - 运行测试脚本`./test.sh`，即可自动加载训练好的模型进行渲染
    ```cmd
    chmod +x ./test.sh
    ./test.sh
    ``` 

- 可视化结果
  - 在终端中输入以下命令，点击 [http://localhost:6006/]() 打开tensorboard进行查看
    ```bash
    tensorboard --logdir ./logs/tensorboard
    ``` 
