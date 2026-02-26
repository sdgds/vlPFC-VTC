# Hierarchical Feedback Model for Occluded Face Processing

基于VTC-vlPFC层级反馈机制的遮挡面孔识别计算模型

---

## 概述

本模型实现了腹侧颞叶皮层(VTC)和腹外侧前额叶皮层(vlPFC)之间的层级反馈交互，用于处理遮挡面孔。模型结合了：
- **自组织映射(SOM)** - 实现拓扑组织
- **随机Hopfield网络** - 实现吸引子动力学
- **自顶向下反馈** - vlPFC对VTC的调制作用

模型在5种遮挡条件下得到验证：完整面孔、遮挡眼睛、上半部、下半部、仅眼睛。

---

## 核心算法流程

### 1. 输入处理
```
原始图像(224×224)
  → AlexNet
  → PCA降维(4维)
  → 归一化
```

### 2. VTC自组织映射
```python
# VTC SOM (200×200神经元)
输入向量(4维) → SOM激活 → 拓扑响应图(200×200)
```
- 使用高斯邻域函数
- 权重归一化
- 前向激活：activation = dot(weights, input)

### 3. Hopfield网络动力学

#### 模式A: 仅循环模式(Recurrent-only)
```
VTC初始状态 → VTC循环动力学 → VTC稳定状态
```

#### 模式B: 反馈模式(Feedback)
```
VTC初始状态
  ↓
VTC循环动力学
  ↓
vlPFC激活(从VTC压缩表征)
  ↓
vlPFC循环动力学
  ↓
vlPFC→VTC反馈投射
  ↓
VTC在反馈调制下继续演化
  ↓
VTC稳定状态
```

### 4. 随机更新规则
```python
# 对于每个神经元i
local_field = Σ(w_ij * s_j) + H_external
```
- `H_external`: 外部场(自底向上输入 + 自顶向下反馈)
- 异步更新：每次随机选择一个神经元更新

### 5. 能量函数
```
E = -0.5 * Σ(w_ij * s_i * s_j)
```
- 网络趋向于最小化能量
- 稳定状态对应能量极小值(吸引子)

---

## 系统要求

- **Python**: 3.8+
- **内存**: 16 GB最低，32 GB推荐
- **存储**: 20 GB
- **GPU**: 可选(仅用于AlexNet特征提取)

---

## 安装

### 1. 创建环境
```bash
conda create -n occluded_face python=3.8
conda activate occluded_face
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 下载预训练权重
需要下载以下文件并放在`Formal/`目录：
- `som_sigma_6.2.npy` - VTC SOM权重(200×200×4)
- `model_VTC_weights.npy` - VTC Hopfield权重(40000×40000, ~12GB)
- `som_vlPFC_weights.npy` - vlPFC SOM权重(20×20×40000)
- `model_vlPFC_weights.npy` - vlPFC Hopfield权重(400×400)
- `face_mask.npy` - 面孔选择性区域掩码
- `object_mask.npy` - 物体选择性区域掩码
- `Data.npy` - AlexNet特征数据
- `mean.npy`, `std.npy` - 归一化参数

---

## 使用方法

### 快速开始

#### 1. 加载模型
```python
import numpy as np
import BrainSOM
import Hopfield_VTCSOM

# 加载VTC SOM
som_VTC = BrainSOM.VTCSOM(200, 200, 4, sigma=6.2, learning_rate=1)
som_VTC._weights = np.load('som_sigma_6.2.npy')

# 加载VTC Hopfield网络
model_VTC = Hopfield_VTCSOM.Stochastic_Hopfield_nn(
    x=200, y=200, pflag=1, nflag=-1,
    patterns=[None, None, None, None]
)
model_VTC._w = np.load('model_VTC_weights.npy')

# 加载vlPFC模型(反馈模式需要)
som_vlPFC = BrainSOM.VTCSOM(20, 20, 40000, sigma=6, learning_rate=1)
som_vlPFC._weights = np.load('som_vlPFC_weights.npy')

model_vlPFC = Hopfield_VTCSOM.Stochastic_Hopfield_nn(
    x=20, y=20, pflag=1, nflag=-1,
    patterns=[None, None, None, None]
)
model_vlPFC._w = np.load('model_vlPFC_weights.npy')
```

#### 2. 运行反馈模式
参见`Occluded_face_formal.ipynb`中的完整实现。

### 主要Notebook文件

1. **Stimuli_formal.ipynb** - 刺激准备
   - AlexNet特征提取
   - GradCAM注意力分析
   - 特征归一化

2. **Occluded_face_formal.ipynb** - 模型仿真
   - VTC-vlPFC层级动力学
   - 5种遮挡条件仿真
   - 结果保存到`model_results/`

3. **Model_results_formal.ipynb** - 结果分析
   - 时间动力学可视化
   - 能量景观分析(UMAP)
   - 解码分析
   - 流形几何度量

---

## 关键参数说明

### SOM参数
- `sigma`: 邻域宽度
- `learning_rate`: 学习率(通常为1.0)
- `neighborhood_function`: 邻域函数类型('gaussian')

### Hopfield参数
- `beta`: 逆温度，控制随机性
  - 高值(100-200): 更确定性，快速收敛
  - 低值(10-50): 更随机，探索更多状态
- `epochs`: 更新步数
  - 仅循环: 80,000步
  - 反馈: 80,000步(vlPFC从50,000步开始)
- `save_inter_step`: 保存间隔(1000步)

### 反馈参数
- `top_down_strength`: 反馈强度(默认: 4)
- `vlPFC_start_time`: vlPFC开始时间(默认: 50,000)

---

## 输出结果

### 1. 动态状态轨迹
- `Dynamic_states_VTC`: VTC随时间的活动(20图像 × 81时间点 × 200 × 200)
- `Dynamic_states_vlPFC`: vlPFC随时间的活动(20图像 × 31时间点 × 20 × 20)

### 2. 能量景观
- UMAP降维的网络状态
- 能量曲面可视化
- 吸引子盆分析

### 3. 性能指标
- **解码准确率**: 面孔vs工具分类(5条件 × 20样本)
- **流形维度**: 内在维度估计
- **流形半径**: 表征的几何扩展

---

## 文件结构

```
Formal/
├── BrainSOM.py                    # SOM实现
├── Hopfield_VTCSOM.py             # 随机Hopfield网络
├── Stimuli_formal.ipynb           # 刺激准备
├── Occluded_face_formal.ipynb     # 主仿真
├── Model_results_formal.ipynb     # 结果可视化
├── Stim_for_model/                # 输入刺激(每条件20张)
│   ├── face/                      # 完整面孔
│   ├── noeye/                     # 遮挡眼睛
│   ├── top_face/                  # 上半部
│   ├── down_face/                 # 下半部
│   └── eyes/                      # 仅眼睛
├── model_results/                 # 输出目录
├── *.npy                          # 预训练权重
├── requirements.txt               # Python依赖
├── README.md                      # 本文件
└── LICENSE                        # MIT许可证
```

---

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件
