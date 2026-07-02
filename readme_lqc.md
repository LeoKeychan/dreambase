### 配置环境
robotwin：
```bash
# Install basic envs and CuRobo
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
bash script/_install.sh

# Download Assets (RoboTwin-OD, Texture Library and Embodiments)
bash script/_download_assets.sh
```

dreamzero：
```bash
# Create conda environment
conda create -n dreamzero python=3.11
conda activate dreamzero

# Install dependencies (PyTorch 2.8+ with CUDA 12.9+)
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129

# Install flash attention
MAX_JOBS=8 pip install --no-build-isolation flash-attn

# [GB200 ONLY, SKIP FOR H100] Install Transformer Engine
pip install --no-build-isolation transformer_engine[pytorch]

# [GB200 ONLY FOR TENSORRT, SKIP FOR H100] Install Tensorrt
pip install tensorrt==10.13.2.6 tensorrt_cu13==10.13.2.6 tensorrt_cu13_libs==10.13.2.6 tensorrt_cu13_bindings==10.13.2.6 --no-deps
pip install transformer_engine==2.10.0 transformer_engine_cu12==2.10.0 transformer_engine_torch==2.10.0
```


### 下载权重
```bash
# 也可以下载完之后都软链接到 dreambase/checkpoints 目录下

# Download Wan2.1 model weights (~28GB)
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P

# Download umt5-xxl tokenizer
hf download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl

# DreamZero-AgiBot (~45GB) 
hf download GEAR-Dreams/DreamZero-AgiBot --repo-type model --local-dir ./checkpoints/DreamZero-AgiBot
```

----------------------------------------------dreambase-------------------------------------------

### 数据处理
```bash
cd xxx/dreambase
conda activate dreamzero
# 转换数据格式，补充生成 metadata
bash scripts/data/convert_robotwin_threeview_tasks.sh
```

### 训练
```bash
cd xxx/dreambase
conda activate dreamzero
bash scripts/train/robotwin_threeview_baseline_training.sh
```

### 评测
```bash
# 开 server
cd xxx/dreambase
conda activate dreamzero
bash eval_utils/server.sh

# 连接 server
cd xxx/RoboTwin/policy/DreamBase
conda activate dreamtwin
# eval.sh 顶部的命令参考 <DREAMBASE_SERVER_PORT> bash eval.sh <task_name> <task_config> <ckpt_setting> <checkpoint_num> <gpu_id>
DREAMBASE_SERVER_PORT=8003 bash eval.sh dump_bin_bigbin demo_clean robotwin_baseline_16train 35000 3
```

----------------------------------------------dreamtriple-------------------------------------------

### 数据处理
```bash
cd xxx/dreamtriple
conda activate robotwin
# scripts/data/render.sh 先根据注释修改路径，然后再渲染 target_front 视角
bash scripts/data/render.sh
# 如果出现某些渲染错误，执行 RETRY_ERRORS=1 scripts/data/render.sh
# 转换数据格式，补充生成 metadata
conda activate dreamzero
bash scripts/data/convert_robotwin_targetview_tasks.sh
```

### 训练
```bash
cd xxx/dreamtriple
conda activate dreamzero
bash scripts/train/robotwin_triple_target_training.sh
```

### 评测
```bash
# 开 server
cd xxx/dreamtriple
conda activate dreamzero
bash eval_utils/server.sh

# 连接 server
cd xxx/RoboTwin/policy/DreamTriple
conda activate dreamtwin
# eval.sh 顶部的命令参考 <DREAMBASE_SERVER_PORT> bash eval.sh <task_name> <task_config> <ckpt_setting> <checkpoint_num> <gpu_id>
DREAMTRIPLE_SERVER_PORT=6010 bash eval.sh click_alarmclock demo_clean robotwin_target_16train 35000 4
```