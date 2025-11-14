# TDSV
A deep learning pipeline for detecting insertion structural variants (SVs) from long-read sequencing alignments.

## 概述
TDSV 提供了一条端到端的数据生成与模型训练/推断流程，旨在在参考基因组与长读长比对文件（BAM/CRAM）上高效、可重复地检测插入型结构变异。该框架包含五个数据生成阶段（Generate_area → Generate_graph → Generate_pic → Generate_tensor → Generate_label），随后进行模型训练（train.py）并在独立数据上进行变异调用（predict.py），最终输出标准 VCF 文件。

该方法强调：
- 面向插入变异的特征抽取与图像/张量表达；
- 基于深度学习的端到端训练与推断；
- 可配置的并行化处理与按染色体/contig 选择分析范围。

## 安装
### 环境与依赖
- Python 3.9
- numpy, pandas, Matplotlib, TensorFlow 2.7, pysam

建议使用 Conda 创建独立虚拟环境以确保依赖隔离与可重复性。

### 1. 创建虚拟环境
```
# 创建
conda create -n TDSV python=3.9
# 激活
conda activate TDSV
# 退出
conda deactivate
```

### 2. 克隆仓库
在激活的虚拟环境中下载并进入 TDSV：
```
git clone https://github.com/DZerTwo/TDSV.git
cd TDSV
```

### 3. 安装依赖
```
conda activate TDSV
# 可使用 conda 或 pip 安装依赖
# 方式一（推荐）：conda
conda install numpy pandas matplotlib tensorflow=2.7 pysam
# 方式二：pip
pip install numpy pandas matplotlib "tensorflow==2.7.*" pysam
```

## 使用方法
TDSV 的标准执行顺序如下：
1) Generate_area
2) Generate_graph
3) Generate_pic
4) Generate_tensor
5) Generate_label
6) train.py（训练得到模型权重）
7) predict.py（生成 VCF 结果）

各阶段的典型命令行与参数说明如下（请根据自身数据路径与资源调整）：

### 1. 生成候选区域（Generate_area）
```
python Generate_area.py bamfile_path_long output_data_folder max_work includecontig
```
- `bamfile_path_long`：长读长数据与参考基因组比对的 BAM/CRAM 文件路径；
- `output_data_folder`：用于存放中间与特征数据的输出目录；
- `max_work`：并行线程数（默认：5）；
- `includecontig`：执行检测的染色体/contig 列表（默认：[]，即使用全部 contig）。

示例：
```
python Generate_area.py ./long_read.bam ./datapath 5 [12,13,14,15,16,17,18,19,20,21,22]
```

### 2. 生成图结构特征（Generate_graph）
```
python Generate_graph.py datapath includecontig
```
- `datapath`：上一步输出的数据目录；
- `includecontig`：同上。

示例：
```
python Generate_graph.py ./datapath [12,13,14,15,16,17,18,19,20,21,22]
```

### 3. 生成图像特征（Generate_pic）
```
python Generate_pic.py datapath includecontig
```
- `datapath`：数据目录；
- `includecontig`：同上。

示例：
```
python Generate_pic.py ./datapath [12,13,14,15,16,17,18,19,20,21,22]
```

### 4. 构建张量输入（Generate_tensor）
```
python Generate_tensor.py datapath includecontig
```
- `datapath`：数据目录；
- `includecontig`：同上。

示例：
```
python Generate_tensor.py ./datapath [12,13,14,15,16,17,18,19,20,21,22]
```

### 5. 生成监督标签（Generate_label）
```
python Generate_label.py datapath includecontig
```
- `datapath`：数据目录；
- `includecontig`：同上。

示例：
```
python Generate_label.py ./datapath [12,13,14,15,16,17,18,19,20,21,22]
```

### 6. 训练模型（train.py）
```
python train.py datapath output_model_path epochs batch_size includecontig
```
- `datapath`：包含张量与标签的数据目录；
- `output_model_path`：训练后模型权重保存路径（如：`./insertion_weights.h5`）；
- `epochs`：训练轮数；
- `batch_size`：批大小；
- `includecontig`：可选，限定参与训练的数据范围。

示例：
```
python train.py ./datapath ./insertion_weights.h5 50 32 [12,13,14,15,16,17,18,19,20,21,22]
```

### 7. 变异调用（predict.py）
```
python predict.py insertion_predict_weight datapath bamfilepath outvcfpath support includecontig
```
- `insertion_predict_weight`：模型权重文件路径；
- `datapath`：用于存放推断所需特征的数据目录（通常与生成阶段一致）；
- `bamfilepath`：长读长比对文件路径；
- `outvcfpath`：输出 VCF 文件路径；
- `support`：最小支持读数阈值；
- `includecontig`：执行检测的染色体/contig 列表（默认：[]，使用全部）。

示例：
```
python predict.py ./insertion_weights.h5 ./datapath ./long_read.bam ./out.vcf 10 [12,13,14,15,16,17,18,19,20,21,22]
```

## 测试数据
下述公开长读长数据可用于评估流程（GRCh37/GRCh38，不同平台）：

### HG002 CLR 数据
```
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/PacBio_MtSinai_NIST/Baylor_NGMLR_bam_GRCh37/HG002_PB_70x_RG_HP10XtrioRTG.bam
```

### HG002 ONT 数据
```
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/UCSC_Ultralong_OxfordNanopore_Promethion/HG002_GRCh37_ONT-UL_UCSC_20200508.phased.bam
```

### HG002 CCS 数据
```
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/PacBio_CCS_15kb/alignment/HG002.Sequel.15kb.pbmm2.hs37d5.whatshap.haplotag.RTG.10x.trio.bam
```

### NA19240 长读长数据
```
http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/hgsv_sv_discovery/working/20160905_smithm_pacbio_aligns/NA19240_bwamem_GRCh38DH_YRI_20160905_pacbio.bam
```

## 输出
- 主要输出：插入变异调用结果（VCF 格式）。
- 建议对输出进行后处理与与已知真值集（如 GIAB benchmarks）比较，以评估召回率与精确率。

## 复现性与可扩展性
- 建议固定随机种子、记录软件版本与硬件配置；
- 可通过 `includecontig` 与 `max_work` 控制空间与计算资源；
- 该流程可扩展至不同平台与参考版本（需相应的比对与预处理）。

## 许可与致谢
- 许可证：请参见仓库中的 LICENSE（如适用）。
- 感谢相关公共数据与工具的维护者（GIAB、PacBio、ONT、pysam、TensorFlow 等）。
