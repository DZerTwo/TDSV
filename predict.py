import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Reshape, multiply, add, Activation
#tf.config.set_visible_devices([], 'GPU')
tf.random.set_seed(123)
from os import listdir
from os.path import isfile, join
from tensorflow import keras
import math
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from collections import Counter
import time
import os
import numpy as np
from tensorflow import keras
from PIL import Image
from tensorflow.keras import layers, models
import pysam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽 INFO 和 WARNING，只显示 ERROR

# 可选 GPU/XLA 配置：先选择可见 GPU，再设置按需内存，再配置混合精度与 XLA
try:
    select_gpu = os.environ.get('SELECT_GPU')  # 例如: '1' 仅使用第1号GPU
    all_gpus = tf.config.list_physical_devices('GPU')
    if all_gpus:
        if select_gpu is not None:
            try:
                idx = int(select_gpu)
                if 0 <= idx < len(all_gpus):
                    tf.config.set_visible_devices(all_gpus[idx], 'GPU')
                    print(f"[INFO] Using only GPU:{idx}")
                else:
                    print(f"[WARN] SELECT_GPU={select_gpu} out of range; using all GPUs")
            except Exception as _e_vis:
                print(f"[WARN] Failed to set visible GPU {select_gpu}: {_e_vis}")

        # 重新获取可见 GPU 列表并开启按需内存
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

        # XLA: 默认关闭，避免首次编译显存激增；可通过 USE_XLA=1 开启
        use_xla = os.environ.get('USE_XLA', '0') == '1'
        try:
            tf.config.optimizer.set_jit(use_xla)
            print(f"[INFO] XLA JIT {'enabled' if use_xla else 'disabled'}.")
        except Exception:
            pass

        # 混合精度：通过 MIXED_PRECISION=1 启用
        if os.environ.get('MIXED_PRECISION', '0') == '1':
            try:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy('mixed_float16')
                print('[INFO] Mixed precision enabled (float16).')
            except Exception:
                pass
    else:
        print('[INFO] No GPU detected by TensorFlow. Training will run on CPU.')
except Exception as _gpu_e:
    print('[WARN] GPU/XLA setup encountered an issue, proceeding with defaults.')

# 补充在本文件内使用到但未定义的工具函数，避免外部依赖导致运行失败
def decode_flag(Flag):
    """将 BAM flag 解码为简单方向编码（与仓库其他脚本保持一致）。"""
    signal = {1 << 2: 0, 1 >> 1: 1, 1 << 4: 2, 1 << 11: 3, 1 << 4 | 1 << 11: 4}
    return signal[Flag] if (Flag in signal) else 0


def c_pos(cigar, refstart):
    """根据 CIGAR 字符串与参考起点，计算(read/ref)起止位置。

    返回: (refstart, refend, readstart, readend)
    """
    number = ''
    numlist = [str(i) for i in range(10)]
    readstart = False
    readend = False
    refend = False
    readloc = 0
    refloc = refstart
    for c in cigar:
        if c in numlist:
            number += c
        else:
            number = int(number)
            if (readstart is False and c in ['M', 'I', '=', 'X']):
                readstart = readloc
            if (readstart is not False and c in ['H', 'S']):
                readend = readloc
                refend = refloc
                break

            if c in ['M', 'I', 'S', '=', 'X']:
                readloc += number

            if c in ['M', 'D', 'N', '=', 'X']:
                refloc += number
            number = ''
    if readend is False:
        readend = readloc
        refend = refloc

    return refstart, refend, readstart, readend

class ECALayer(tf.keras.layers.Layer):
  
    def __init__(self):
        super(ECALayer, self).__init__()



    def build(self, input_shape):
    
       
        self.in_channel = input_shape[-1]
        self.kernel_size = int(abs((math.log(self.in_channel, 2) +1 ) / 2))

        
        if self.kernel_size % 2:
            self.kernel_size = self.kernel_size

        
        else:
            self.kernel_size = self.kernel_size + 1
        self.con = tf.keras.layers.Conv1D(filters=1, kernel_size=self.kernel_size, padding='same', use_bias=False)   
    def call(self, inputs):
        
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    
        x = tf.keras.layers.Reshape(target_shape=(self.in_channel, 1))(x)

        x = self.con(x)

        x = tf.nn.sigmoid(x)

        x = tf.keras.layers.Reshape((1,1,self.in_channel))(x)

        output = tf.keras.layers.multiply([inputs, x])
        return output
    def compute_output_shape(self, input_shape):
        return input_shape


# In[4]:


def cbam_block(cbam_feature, ratio=7,kernel_size = (1,5)):

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature,kernel_size)
    return cbam_feature

def channel_attention(input_feature, ratio=8):

    #channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[-1]
    filters = max(1, int(channel//ratio))
    shared_layer_one = tf.keras.layers.Dense(filters,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = tf.keras.layers.Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_feature)    
    avg_pool = tf.keras.layers.Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_feature)
    max_pool = tf.keras.layers.Reshape((1,1,channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
   

    cbam_feature = tf.keras.layers.Add()([avg_pool,max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)


    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature,kernel_siz):
    kernel_size = kernel_siz

    channel = input_feature.shape[-1]
    cbam_feature = input_feature

    avg_pool = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    #assert avg_pool._keras_shape[-1] == 1
    max_pool = tf.keras.layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    #assert max_pool._keras_shape[-1] == 1
    concat = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
    #assert concat._keras_shape[-1] == 2
    cbam_feature = tf.keras.layers.Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)	
    #assert cbam_feature._keras_shape[-1] == 1

    return multiply([input_feature, cbam_feature])


# In[5]:


def cnn_model():
    
    inputs = tf.keras.Input(shape=(200,5,1))
    x = tf.keras.layers.Conv2D(128, kernel_size =(2,5), padding='same', activation='elu')(inputs)
    x = tf.keras.layers.MaxPool2D((2,1))((x))
    x = tf.keras.layers.SeparableConv2D(64,kernel_size =(2,1), padding='same', activation='elu')(x)
    x = tf.keras.layers.MaxPool2D((2,1))(x)
    x = cbam_block(x)
    x = tf.keras.layers.SeparableConv2D(64,kernel_size =(2,1), padding='same', activation='elu')(x)
    x = tf.keras.layers.MaxPool2D((2,1))(x)

    x = tf.keras.layers.SeparableConv2D(64,kernel_size =(2,1), padding='same', activation='elu')(x)
    x = tf.keras.layers.MaxPool2D((2,1))(x)
    x = ECALayer()(x)
    x = tf.keras.layers.SeparableConv2D(64,kernel_size =(2,1), padding='same', activation='elu')(x)
    x = tf.keras.layers.MaxPool2D((2,1))(x)

    x = tf.keras.layers.SeparableConv2D(64,kernel_size =(2,1), padding='same', activation='elu')(x)
    x = tf.keras.layers.MaxPool2D((2,1))(x)
    x = ECALayer()(x)
    x = tf.keras.layers.SeparableConv2D(64,kernel_size =(2,1), padding='same', activation='elu')(x)
    x = tf.keras.layers.MaxPool2D((2,1))(x)
    x = tf.keras.layers.Flatten()(x)
    model = tf.keras.Model(inputs, x)
    print(f'cnn_model = {model.summary()}')
    return model


# In[6]:

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
# 可选更强的轻量级骨干
try:
    from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
    # 为 EfficientNetB0 提供与 ImageNet 预训练权重匹配的预处理
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
except Exception:
    # 如果你的 TF 版本不包含这些模块，这里只是保底，不影响原有轻量模型
    MobileNetV2 = None
    EfficientNetB0 = None
    efficientnet_preprocess = None

class MultiplyLayer(Layer):
    def __init__(self, **kwargs):
        super(MultiplyLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        A, B = inputs
        return tf.multiply(A, B)
    
    def get_config(self):
        return super(MultiplyLayer, self).get_config()


class AddPositionalEmbedding(layers.Layer):
    """学习型位置编码: 对输入序列按时间维添加 Embedding 形式的位置向量。

    使用方法: x = AddPositionalEmbedding(max_len=2048)(x)
    其中 x 的形状为 (batch, time, dim)。
    """
    def __init__(self, max_len=2048, d_model=None, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model  # 可显式指定，否则在 build 中从输入推断

    def build(self, input_shape):
        feat_dim = int(self.d_model or input_shape[-1])
        self.pos_embedding = layers.Embedding(
            input_dim=self.max_len,
            output_dim=feat_dim,
            name='pos_embedding'
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, D)
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_emb = self.pos_embedding(positions)          # (T, D)
        pos_emb = tf.expand_dims(pos_emb, axis=0)        # (1, T, D)
        return x + pos_emb
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model,
        })
        return config


def create_lightweight_model(input_shape=(200, 200, 3), output_dim=100):
    """创建超轻量级CNN模型，参数量<0.5M"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # 初始卷积层 (快速下采样)
    x = layers.Conv2D(16, 7, strides=2, padding='same')(inputs)  # 200x200 -> 100x100
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 深度可分离卷积块1
    x = depthwise_separable_block(x, 32, strides=2)  # 100x100 -> 50x50
    
    # 深度可分离卷积块2
    x = depthwise_separable_block(x, 64, strides=2)  # 50x50 -> 25x25
    
    # 深度可分离卷积块3
    x = depthwise_separable_block(x, 128, strides=2)  # 25x25 -> 13x13
    
    # 深度可分离卷积块4 (无下采样)
    x = depthwise_separable_block(x, 256, strides=1)
    
    # 全局平均池化
    x = layers.GlobalAveragePooling2D()(x)
    
    # 输出层
    outputs = layers.Dense(output_dim)(x)
    
    return models.Model(inputs, outputs)

def depthwise_separable_block(x, filters, strides=1):
    """深度可分离卷积块"""
    # 深度卷积
    x = layers.DepthwiseConv2D(3, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 逐点卷积
    x = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    return x

# 创建模型
model = create_lightweight_model()

# 打印模型摘要
model.summary()


def create_mobilenetv2_backbone(input_shape=(200,200,3), output_dim=128, use_pretrained=False):
    """使用 MobileNetV2 作为图像分支骨干（轻量但效果好）

    Parameters:
        input_shape: 图像输入形状
        output_dim: 输出特征向量维度
        use_pretrained: 是否加载 ImageNet 权重（如果可用）
    """
    if MobileNetV2 is None:
        raise RuntimeError("MobileNetV2 not available in this TF build")

    weights = 'imagenet' if use_pretrained else None
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights=weights, pooling='avg')
    inputs = base.input
    x = base.output
    x = layers.Dense(output_dim, activation=None, name='mobilenetv2_proj')(x)
    return Model(inputs, x, name='mobilenetv2_backbone')


def create_efficientnetb0_backbone(input_shape=(200,200,3), output_dim=128, use_pretrained=False, freeze_base=True):
    """使用 EfficientNetB0 作为图像分支骨干（较强的轻量模型，带预处理与尺寸自适配）。

    参数:
        input_shape: 输入图像尺寸，默认 (200,200,3)
        output_dim: 输出特征维度
        use_pretrained: 是否加载 ImageNet 预训练权重
        freeze_base: 是否冻结 EfficientNet 主干参数（微调时推荐先冻结）
    """
    if EfficientNetB0 is None:
        raise RuntimeError("EfficientNetB0 not available in this TF build")

    inputs = layers.Input(shape=input_shape, name='effb0_input')
    x = inputs
    # EfficientNetB0 的 ImageNet 预训练通常以 224x224 分辨率训练；这里自动调整
    if input_shape[0] != 224 or input_shape[1] != 224:
        x = layers.Resizing(224, 224, name='effb0_resize')(x)

    # 仅在使用预训练时应用官方预处理，否则保持与原数据一致
    if use_pretrained and efficientnet_preprocess is not None:
        x = layers.Lambda(efficientnet_preprocess, name='effb0_preprocess')(x)

    weights = 'imagenet' if use_pretrained else None
    # 单独实例化 base，再作为层调用，便于自定义输入流水线
    base = EfficientNetB0(include_top=False, weights=weights, pooling='avg')
    if use_pretrained and freeze_base:
        base.trainable = False

    x = base(x)
    x = layers.Dense(output_dim, activation=None, name='efficientnetb0_proj')(x)
    return Model(inputs, x, name='efficientnetb0_backbone')


def create_multimodal_model(num_classes=1, dropout_rate=0.4, image_backbone='light', image_output_dim=128, use_pretrained=False,
                            temporal_model='transformer', transformer_layers=2, transformer_heads=4, transformer_dropout=0.1):
    # 双模态输入
    video_input = layers.Input(shape=(None, 200, 5, 1))
    image_input = layers.Input(shape=(None, 200, 200, 3))
    
    cnn_layer = cnn_model()
    video_features = layers.TimeDistributed(cnn_layer)(video_input)
    
    # 图像特征提取
    def time_distributed_picture():
        inputs = layers.Input(shape=(200, 200, 3))
        # 原始代码引用的 picture() 在当前文件未实现，仅在其它脚本中出现注释占位。
        # 这里直接返回输入的轻量卷积骨干以保持兼容。
        outputs = inputs
        return Model(inputs, outputs, name='identity_picture')
    
    # image_model = time_distributed_picture()
    # 支持可选的图像骨干：保留原轻量模型，或使用 MobileNetV2 / EfficientNetB0
    if image_backbone == 'mobilenetv2':
        image_model = create_mobilenetv2_backbone(output_dim=image_output_dim, use_pretrained=use_pretrained)
    elif image_backbone == 'efficientnetb0':
        # 使用带预处理与尺寸自适配的 EfficientNetB0 预训练骨干
        image_model = create_efficientnetb0_backbone(output_dim=image_output_dim, use_pretrained=use_pretrained, freeze_base=True)
    else:
        # 默认保留原有轻量模型（参数化输出维度）
        try:
            image_model = create_lightweight_model(input_shape=(200,200,3), output_dim=image_output_dim)
        except TypeError:
            # 兼容原来没有参数的实现
            image_model = create_lightweight_model()

    image_features = layers.TimeDistributed(image_model)(image_input)
    
    # 双向跨模态注意力
    # video -> image 与 image -> video 两个方向的注意力，并拼接原始特征与两个注意力结果
    def bidirectional_cross_modal_attention(video_feat, image_feat, hidden_dim=128):
        # 方向1: video 作为 query, image 作为 key/value
        v_query = layers.Dense(hidden_dim, name='video_query_dense')(video_feat)
        v_query = layers.LayerNormalization(name='video_query_ln')(v_query)
        i_key   = layers.Dense(hidden_dim, name='image_key_dense')(image_feat)
        i_value = layers.Dense(hidden_dim, name='image_value_dense')(image_feat)
        attended_image = layers.Attention(name='video_to_image_attn')([v_query, i_value, i_key])

        # 方向2: image 作为 query, video 作为 key/value
        i_query = layers.Dense(hidden_dim, name='image_query_dense')(image_feat)
        i_query = layers.LayerNormalization(name='image_query_ln')(i_query)
        v_key   = layers.Dense(hidden_dim, name='video_key_dense')(video_feat)
        v_value = layers.Dense(hidden_dim, name='video_value_dense')(video_feat)
        attended_video = layers.Attention(name='image_to_video_attn')([i_query, v_value, v_key])

        # 门控融合：对两个注意力方向进行加权求和（权重通过 softmax 学习，随时间步变化）
        concat_feats = layers.Concatenate(axis=-1, name='gate_concat')([video_feat, attended_image, attended_video])
        gate_logits = layers.Dense(2, name='gate_logits')(concat_feats)            # (..., 2)
        gate = layers.Softmax(axis=-1, name='gate_softmax')(gate_logits)           # (..., 2)
        g_img = layers.Lambda(lambda t: t[..., 0:1], name='gate_img')(gate)        # (..., 1)
        g_vid = layers.Lambda(lambda t: t[..., 1:2], name='gate_vid')(gate)        # (..., 1)

        weighted_attended = layers.Add(name='attended_weighted_sum')([
            layers.Multiply(name='mul_img')([g_img, attended_image]),
            layers.Multiply(name='mul_vid')([g_vid, attended_video])
        ])  # (..., hidden_dim)

        # 与原 video 特征拼接（保持与原单向注意力类似的维度增长：Dv + hidden_dim）
        fused = layers.Concatenate(axis=-1, name='gated_fused')([video_feat, weighted_attended])
        return fused

    fused_features = bidirectional_cross_modal_attention(video_features, image_features)
    
    # 时序建模（支持 Transformer 或 BiGRU）
    def transformer_encoder(x, num_heads=4, key_dim=64, ff_dim=256, dropout=0.1, name=None):
        y = layers.LayerNormalization(name=f'{name}_ln1' if name else None)(x)
        y = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=f'{name}_mha' if name else None)(y, y)
        y = layers.Dropout(dropout, name=f'{name}_drop1' if name else None)(y)
        x = layers.Add(name=f'{name}_add1' if name else None)([x, y])

        y = layers.LayerNormalization(name=f'{name}_ln2' if name else None)(x)
        y = layers.Dense(ff_dim, activation='relu', name=f'{name}_ff1' if name else None)(y)
        y = layers.Dropout(dropout, name=f'{name}_drop2' if name else None)(y)
        y = layers.Dense(x.shape[-1], name=f'{name}_ff2' if name else None)(y)
        y = layers.Dropout(dropout, name=f'{name}_drop3' if name else None)(y)
        x = layers.Add(name=f'{name}_add2' if name else None)([x, y])
        return x

    if temporal_model == 'transformer':
        x = fused_features
        # 学习型位置编码（避免在模型外部使用 tf.shape 触发 KerasTensor 限制）
        d_model = fused_features.shape[-1]
        if d_model is not None:
            x = AddPositionalEmbedding(max_len=2048, d_model=int(d_model), name='add_positional_embedding')(x)
        else:
            x = AddPositionalEmbedding(max_len=2048, name='add_positional_embedding')(x)

        # 计算合理的 key_dim 与 ff_dim（基于静态 D 值；若未知则使用回退值）
        d_model_int = int(d_model) if d_model is not None else 256
        key_dim = max(32, d_model_int // transformer_heads)
        ff_dim = max(128, d_model_int * 4)

        for i in range(transformer_layers):
            x = transformer_encoder(x, num_heads=transformer_heads, key_dim=key_dim, ff_dim=ff_dim, dropout=transformer_dropout, name=f'trans_enc_{i}')
    else:
        # 回退到原来的 BiGRU 堆叠
        x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(fused_features)
        x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    
    # 分类头
    x = layers.Dense(256)(x)
    for units in [128, 64, 32]:
        x = layers.Dense(units, kernel_regularizer=l2(1e-4))(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=[video_input, image_input], outputs=outputs)
    
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        metrics=[
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'), 
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model
if os.environ.get("QUICK_RUN") == "1":
    # 轻量构建一次模型并打印结构，避免在默认执行时触发重量级下载/训练
    _quick_model = create_multimodal_model()
    _quick_model.summary()

class DataGenerator(keras.utils.Sequence):
    def __init__(self, X_images_files, X_tensors, y, batch_size=80, shuffle=False ,**kwargs):
        """
        多模态数据生成器（支持图像 + 张量输入）
        :param X_images: Python 列表，存储 (100, 200, 200, 3) 的 NumPy 数组
        :param X_tensors: Python 列表，存储 (time_steps, 200, 5, 1) 的 NumPy 数组
        :param y: 标签列表
        :param batch_size: 每批数据量（默认 80）
        :param shuffle: 是否在每个 epoch 后打乱数据顺序
        """
        super().__init__(**kwargs) 
        self.X_images_files = X_images_files  
        self.X_tensors = X_tensors  
        self.y = np.array(y)  
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X_images_files))  # 初始化索引数组
        self.on_epoch_end()

    def __len__(self):
        """返回每个 epoch 的批次数"""
        return int(np.ceil(len(self.X_images_files) / self.batch_size))

    def __getitem__(self, index):
        """生成一个批次的数据（动态加载，避免内存爆炸）"""
        # 获取当前批次的索引
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.X_images_files))
        batch_indices = self.indices[start_idx:end_idx] # 例如 [0, 1, 2, ..., 79]

        # 动态加载当前批次的图像和张量数据
        batch_images_path = [self.X_images_files[i] for i in batch_indices] # 仅转换当前批次
        batch_images = read_images_to_array(batch_images_path)
        batch_tensors = np.array([self.X_tensors[i] for i in batch_indices])  # 仅转换当前批次
        batch_tensors = batch_tensors.reshape(batch_tensors.shape[0], 100, 200, 5, 1)  # 确保形状为 (batch_size, time_steps, 200, 5, 1)
        batch_y = self.y[batch_indices]

        # 如果启用混合精度，优先在数据侧将输入转换为 float16，降低显存占用
        if os.environ.get('MIXED_PRECISION', '0') == '1':
            try:
                batch_images = batch_images.astype(np.float16, copy=False)
                batch_tensors = batch_tensors.astype(np.float16, copy=False)
            except Exception:
                pass

        # 返回多模态输入（图像 + 张量）和标签
        return (batch_tensors, batch_images), batch_y

    def on_epoch_end(self):
        """每个 epoch 结束时打乱数据顺序"""
        if self.shuffle:
            np.random.shuffle(self.indices)

def read_images_to_array(folder_paths_list):
    # 初始化一个空列表来存储图像数据
    images_batch = []
    for folder_paths in folder_paths_list:
        images = []
        for folder_path in folder_paths:
            if folder_path.endswith('.png'):
                with Image.open(folder_path) as img:
                    # 将图像转换为RGB模式，以防它是其他模式（如RGBA）
                    img = img.convert('RGB')
                    
                    # 将图像数据转换为numpy数组
                    img_array = np.array(img)
                    
                    # 确保图像是200x200x3
                    if img_array.shape == (200, 200, 3):
                        images.append(img_array)
                    else:
                        print(f"Image {folder_path} does not have the expected shape (200x200x3).")
            else:
                white_pics = np.full((200,200,3),fill_value=255,dtype=np.uint8)
                images.append(white_pics)  # 如果不是PNG文件，填充空白图像
        images_array = np.array(images)
        images_batch.append(images_array)
    return np.array(images_batch)


def load_data(pic_path, tensor_path, labels_path, chromosomes):
    """加载指定染色体范围的数据"""
    label_files = sorted(os.listdir(labels_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
    tensor_files = os.listdir(tensor_path)
    
    all_tensors = []
    all_pics_files = []
    all_labels = []

    for i in chromosomes:
        # 提取标签数据
        label_file_path = os.path.join(labels_path, label_files[i-1])
        label_list = []
        with open(label_file_path, 'r') as f:
            for line in f:
                label_list.append(line.strip())
        
        # 提取tensor数据
        chrom_tensor_files = [f for f in tensor_files if f.startswith(f"{i}_") and f.endswith('.npy')]
        chrom_tensor_files = sorted(chrom_tensor_files, key=lambda x: int(x.split('_')[1]))
        tensors_list = []
        for tensor_file in chrom_tensor_files:
            tensor_file_path = os.path.join(tensor_path, tensor_file)
            arr = np.load(tensor_file_path)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1000)
            tensors_list.append(arr)
        tensors = np.vstack(tensors_list)
        labels = np.array(label_list, dtype=np.int32)
        
        if tensors.shape[0] != labels.shape[0]:
            raise ValueError(f"染色体 {i}: tensors数量 {tensors.shape[0]} 与标签数量 {labels.shape[0]} 不匹配")
        
        # 提取图片数据
        pics_data_path = os.path.join(pic_path, f"chr{i}_pics")
        images_files = [os.path.join(pics_data_path, f) for f in os.listdir(pics_data_path) if f.endswith('.png')]
        
        if len(images_files) != labels.shape[0]:
            raise ValueError(f"染色体 {i}: 图片数量 {len(images_files)} 与标签数量 {labels.shape[0]} 不匹配")
        
        all_pics_files.append(images_files)
        all_tensors.append(tensors)
        all_labels.append(labels)

    # 拼接数据
    all_pics_files = np.concatenate(all_pics_files, axis=0)
    all_tensors = np.concatenate(all_tensors, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 按batch_size=100分批
    batch_size = 100
    num_samples = all_tensors.shape[0]
    num_full_batches = num_samples // batch_size
    remainder = num_samples % batch_size

    pics_batches = [all_pics_files[j:j+batch_size] for j in range(0, num_full_batches * batch_size, batch_size)]
    tensors_batches = [all_tensors[j:j+batch_size] for j in range(0, num_full_batches * batch_size, batch_size)]
    label_batches = [all_labels[j:j+batch_size] for j in range(0, num_full_batches * batch_size, batch_size)]

    if remainder > 0:
        pad_pics = np.zeros((batch_size,), dtype=all_pics_files.dtype)
        pad_pics[:remainder] = all_pics_files[-remainder:]
        pics_batches.append(pad_pics)
        
        pad_tensors = np.zeros((batch_size, all_tensors.shape[1]), dtype=all_tensors.dtype)
        pad_labels = np.zeros((batch_size,), dtype=all_labels.dtype)
        pad_tensors[:remainder] = all_tensors[-remainder:]
        pad_labels[:remainder] = all_labels[-remainder:]
        
        tensors_batches.append(pad_tensors)
        label_batches.append(pad_labels) 
    
    return tensors_batches, pics_batches, label_batches

def train(model_path, pic_path, tensor_path, labels_path, main_path,
          image_backbone='light', image_output_dim=128, use_pretrained=False,
          epochs=300):
    # 加载训练集 (1-10号染色体)
    train_tensors, train_pics, train_labels = load_data(
        pic_path, tensor_path, labels_path, chromosomes=range(1, 11)
    )
    
    # 加载验证集 (11号染色体)
    val_tensors, val_pics, val_labels = load_data(
        pic_path, tensor_path, labels_path, chromosomes=[11]
    )
    
    # 创建数据生成器
    batch_size = int(os.environ.get('BATCH_SIZE', '4'))  # 默认显存友好
    print(f"[INFO] Using batch_size={batch_size}")
    train_generator = DataGenerator(
        X_images_files=train_pics,
        X_tensors=train_tensors,
        y=train_labels,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = DataGenerator(
        X_images_files=val_pics,
        X_tensors=val_tensors,
        y=val_labels,
        batch_size=max(1, batch_size//2),
        shuffle=False  # 验证集不需要打乱
    )

    # 初始化模型
    model = create_multimodal_model(image_backbone=image_backbone,
                                    image_output_dim=image_output_dim,
                                    use_pretrained=use_pretrained)

    # 训练模型，添加验证集
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(main_path, "model_fution_best_val.weights.h5"),  # 区分文件名
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',  # 监控验证损失
            mode='min'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',  # 监控验证损失
            patience=6,         # 增加耐心值
            restore_best_weights=True,
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',  # 监控验证损失
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.CSVLogger('training_log.csv')  # 添加训练日志
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,  # 添加验证集
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # 保存最终模型权重
    model.save_weights(model_path)
    print("训练完成，模型权重已保存至:", model_path)
    
    # 评估模型在验证集上的表现
    print("\n验证集评估结果:")
    eval_results = model.evaluate(val_generator)
    print(f"验证损失: {eval_results[0]:.4f}, 准确率: {eval_results[5]:.4f}, "
          f"精确率: {eval_results[6]:.4f}, 召回率: {eval_results[7]:.4f}")




def ins_signature(pre,bamfile):

    data = []
    for chr_name,start,end in pre:

        for read in bamfile.fetch(chr_name,start,end):
            aligned_length = read.reference_length
            if aligned_length == None:
                aligned_length= 0
            if (read.mapping_quality >= 0) and aligned_length >= 0:
                cigar = []
                sta = read.reference_start
                for ci  in read.cigartuples:
                    
                    #print(start)
                    if ci[0] in [0, 2, 3, 7, 8]:
                        sta += ci[1]
                    elif ci[0] == 1 :
                       #print(ci[1])
                        if ci[1] >=50 and (abs(sta-start) < 200):
                            cigar.append([sta,sta,ci[1]])
                        #sta += 1
                if len(cigar) !=0:
                    cigar = np.array(cigar)
                    #print(cigar)
                    cigar = cigar[np.argsort(cigar[:,0])]
                    a = mergecigar(cigar)
                    #print(a)
                    data.extend(a)
            #data = np.array(data)
            #print(data)
            #insloc_2(AlignedSegment.reference_start, start, end, AlignedSegment.cigartuples, delinfo, insinfo)
            if(read.has_tag('SA') == True):
                code = decode_flag(read.flag)
                sapresent = True
                rawsalist = read.get_tag('SA').split(';')
                #print(rawsalist)
                for sa in rawsalist[:-1]:
                    sainfo = sa.split(',')
                    #print(sainfo)
                    tmpcontig, tmprefstart, strand, cigar = sainfo[0], int(sainfo[1]), sainfo[2], sainfo[3]
                    if(tmpcontig != chr_name):
                        continue
                    #print(code,strand)   
                    if((strand == '-' and (code %2) ==0) or (strand == '+' and (code %2) ==1)):
                        refstart_1, refend_1, readstart_1, readend_1 =  read.reference_start, read.reference_end,read.query_alignment_start,read.query_alignment_end
                        refstart_2, refend_2, readstart_2, readend_2 = c_pos(cigar, tmprefstart)
                        a = readend_1 - readstart_2
                        b = refend_1 - refstart_2
                        if(abs(b-a)<30):
                            continue
                        #print(b-a)
                        #if(abs(b)<2000):
                        if((b-a)>=50 and ((b-a)>0)):
                            #print(refstart_2,b-a)
                            data22 = []
                            
                            if(refend_1<=end and refend_1>=start):
                                data22.append([refend_1,refend_1,abs((b-a))])

                            if(refstart_2<=end and refstart_2>=start):
                                data22.append([refstart_2,refstart_2,abs((b-a))])
                            #print(data22)
                            #data22.append([(refend_1+ refstart_2)//2,(refend_1+ refstart_2)//2, abs(b-a)])
                            #print(data22)
                            data22 = np.array(data22)
                            #print(data1)
                            if len(data22)==0:
                                continue
                            data.extend(data22)
            #print(len(data))

    data = np.array(data)
    #print(len(data))  
    
    if len(data) == 0:
        return data
    #print(data.shape)
    data = data[np.argsort(data[:,0])]
            
                
    return data


def mergecigar(infor):
    data = []
    i = 0
    while i>=0:
        count = 0
        if i >(len(infor)-1):
            break
        lenth = infor[i][2]
        for j in range(i+1,len(infor)):
            #print(i,j)
            if abs(infor[j][1] - infor[i][1]) <= 40: 
                count = count + 1
                infor [i][1] = infor[j][0]#改[0]0
                lenth = lenth +  infor[j][2] #+ abs(infor[j][0] - infor[i][0])
        

        #print(infor)
        data.append([infor[i][0],infor[i][0]+1, lenth])


        if count == 0:
            i += 1
        else :
            i += (count+1)
    return data

def merge(infor):
    data = []
    i = 0
    while i>=0:
        dat = []
        
        count = 0
        if i >(len(infor)-1):
            break
        dat.append(infor[i])
        for j in range(i+1,len(infor)):
            #print(i,j)
            if( (abs(infor[i][0] -infor[j][0]) <= 1500) and (abs(infor[i][1] - infor[j][1])<= 1500)):
                count = count + 1
                dat.append(infor[j])
        #print(infor[i],count)
        dat = np.array(dat)
        data.append(dat)


        if count == 0:
            i += 1
        else :
            i += (count+1)
    return data


def mergedeleton_long(pre,index,chr_name,bamfile):
    data = []
    insertion = []
    #print(len(pre),index.shape)
    for i in range(len(pre)):
        if pre[i] > 0.5:
            data.append([chr_name,index[i]*200,index[i]*200+200])
    signature = ins_signature(data,bamfile)
    #for dd in signature:
    #    print(dd)
    dell = merge(signature)
    #print(dell)
    for sig in dell:
            pp = np.array(sig)
            #print(sig)
            start = math.ceil(np.median(pp[:,0]))
            kk = int(len(pp)/2)
            svle = np.sort(pp[:,2])
            #print(svle)
            #en = math.ceil(pp[:1].mean)
            length = math.ceil(np.median(svle[kk:]))
            #print(kk,start,length)
            insertion.append([chr_name,start,length,len(pp),'INS'])
    return insertion 


# In[8]:


def batchdata(data,pic_data,batch_size,step,window = 200):
    data = data[:,:]
    pic_data = pic_data[:,:]
    if(data.shape[0] != pic_data.shape[0]):
        print("data and pic_data not match")
        raise ValueError("data and pic_data not match")
    #data = data.reshape(-1,8)
    #data[:,-3:]=0
    #print(data)
    #print(data.shape)
    if step != 0:
        data = data.reshape(-1,5)[step:(step - window)] # data.shape=(3004,1000) -> (3004*200,5)[100:-100]
        pic_data = pic_data.reshape(-1,200,3)[step:(step - window)] # pic_data.shape=(3004,200,200,3) -> (3004*200,200,3)[100:-100]
        
    data = data.reshape(-1,200,5)
    pic_data = pic_data.reshape(-1,200,200,3)

    size = data.shape[0]//batch_size
    size_ = data.shape[0]%batch_size
    #print(data.shape[0],size_)
    return data[:size*batch_size],data[size*batch_size:],pic_data[:size*batch_size],pic_data[size*batch_size:]
def predcit_step(base,predict):
    for i in range(len(predict)):
        if predict[i] >= 0.5:
            base[i] = 1
            base[i+1] = 1
    return base    


import numpy as np
import math
from tensorflow.keras.utils import Sequence

class load_test(Sequence):

    def __init__(self, x_y_set, batch_size):
        self.x_y_set = x_y_set
        self.x = self.x_y_set[:,:]
        self.batch_size = batch_size

    def __len__(self):
        return math.floor(len(self.x) / self.batch_size)

        
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x = batch_x.reshape(-1,100,200,5,1)
        
        
        return np.array(batch_x)


# In[9]:


def find_geno(inss,b):
    count = 0
    data = []
    geno_count = []
    sss = time.time()
    for i in inss:
        contig,start,end,support,svlen = str(i[0]),int(i[1]),(int(i[2])),10,int(i[5])

        start_ge = (start//200)*200 
        end_ge = (end//200+1)*200

        c = b[b[:,0]==contig]
        #print(c)

        w = np.where((np.array(c[:,1],dtype=int)>= start_ge)&(np.array(c[:,2],dtype=int)<= end_ge) )[0]
        print(contig,start_ge,end_ge,c[w][0])
        if int(c[w][0][-1]) == 1:
            data.append([contig,start,svlen,support,'INS','1/1'])
        else:
            data.append([contig,start,svlen,support,'INS','0/1'])

        #geno_count = np.array(c[w][:,-1],dtype = int)
        #print(geno_count)
        #geno_count = np.array(geno_count)
    return data


# In[10]:

def tovcf(rawsvlist, contig2length, outputpath):
    head = """##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">\n"""
    body = ''
    for contig in contig2length:
        body += "##contig=<ID=" + contig + ", length=" + str(int(contig2length[contig])) + ">\n"
    tail = """##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the structural variant">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of SV:DEL=Deletion, INS=Insertion">
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description="Number of reads supporting the variant">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t.\n"""
    vcfinfo = head + body + tail
    ins_id = 0
    for rec in rawsvlist:
        contig = rec[0]
        geno = rec[-1]
        recinfo = 'SVLEN=' + str(int(rec[2])) + ';SVTYPE=' + 'INS' + ';END=' + str(rec[1]) + ';SUPPORT=' + str(rec[3]) + '\tGT\t' + str(
            geno) + '\n'
        vcfinfo += (contig + '\t' + str(
            int(rec[1])) + '\t' + 'INS.'+ str(ins_id)  + '\t' + 'N' + '\t' + '<INS>' + '\t' + str(int(rec[3])+1) + '\t' + 'PASS' + '\t' + recinfo)
        ins_id += 1
    f = open(outputpath, 'w')
    f.write(vcfinfo)
    f.close()


# In[49]:


def predict_funtion(area_list_path,ins_predict_weight,datapath,pics_path,bamfilepath,outvcfpath,contigg,support):
    start = 0
    contig2length = {}
    bamfile = pysam.AlignmentFile(bamfilepath,'rb')
    
    if len(contigg) == 0:
        contig = []
        for count in range(len(bamfile.get_index_statistics())):
            contig.append(bamfile.get_index_statistics()[count].contig)
            contig2length[bamfile.get_index_statistics()[count].contig] = bamfile.lengths[count]
    else:
        contig = np.array(contigg).astype(str)
    for count in range(len(bamfile.get_index_statistics())):
            contig2length[bamfile.get_index_statistics()[count].contig] = bamfile.lengths[count]
    
    resultlist = [['CONTIG', 'START', 'SVLEN', 'READ_SUPPORT', 'SVTYPE']]
    
    # 按训练一致的骨干与预训练配置构建模型（从环境变量读取，默认与训练脚本一致）
    pred_backbone = os.environ.get('IMAGE_BACKBONE', 'light')
    pred_use_pretrained = os.environ.get('USE_PRETRAINED', '0') == '1'
    print(f"[INFO] 预测阶段使用骨干: backbone={pred_backbone}, pretrained={pred_use_pretrained}")

    predict_ins = create_multimodal_model(
        image_backbone=pred_backbone,
        image_output_dim=128,
        use_pretrained=pred_use_pretrained
    )

    # 创建虚拟输入数据（更可靠的构建方式）
    dummy_data_1 = np.zeros((1, 100, 200, 5, 1))  # 保持原始维度，batch_size=1
    dummy_data_2 = np.zeros((1, 100, 200, 200, 3))
    if os.environ.get('MIXED_PRECISION', '0') == '1':
        try:
            dummy_data_1 = dummy_data_1.astype(np.float16)
            dummy_data_2 = dummy_data_2.astype(np.float16)
        except Exception:
            pass

    # 通过 predict 触发完整构建
    _ = predict_ins.predict([dummy_data_1, dummy_data_2], verbose=0)

    # 然后加载权重（可通过 SKIP_MISMATCH=1 放宽部分不匹配，但不建议常用）
    skip_mismatch = os.environ.get('SKIP_MISMATCH', '0') == '1'
    predict_ins.load_weights(ins_predict_weight, skip_mismatch=skip_mismatch)
    
    predict_ins.summary()
    # dummy_input = np.zeros((80, 100, 200, 5, 1))  # 输入形状
    # dummy_input_pic = np.zeros((80, 100, 200, 200, 3))  # 输入形状
    # predict_ins.predict([dummy_input, dummy_input_pic])  # 触发模型构建
    
    # 预测批大小（默认与训练一致，或通过 BATCH_SIZE_PRED 单独控制）
    pred_bs = int(os.environ.get('BATCH_SIZE_PRED', os.environ.get('BATCH_SIZE', '4')))

    for ww in contig:
        pics_list = os.listdir(f"{pics_path}/chr{ww}_pics")
        chr_name = ww
        chr_length = contig2length[ww]
        if 'chr' in chr_name:
            chr_name1 = chr_name[3:]
        else:
            chr_name1 = chr_name
        ider = math.ceil(chr_length / 10000000)
        start = 0
        print('+++++++',chr_name,'++++++++++')
        print('chr:',chr_name,ider)
        
        with open(area_list_path + '/' + "area_list_" + chr_name + '.txt', 'r') as f:
            area_list = [int(line) for line in f]
        
        for i in range(ider):
            print('insertion_predict_chr:',chr_name,i,'/',ider)
            try:
                #print(chr_name1)
                pics_selectied = [pic for pic in pics_list if int(pic.split('_')[1].split('.')[0])*200 >= start and int(pic.split('_')[1].split('.')[0])*200 < start + 10000000]
                if len(pics_selectied) == 0:
                    print("No pics found for:", chr_name1, start, start + 10000000)
                # 将pics_selectied列表存储的文件名读取为一整个numpy数据
                if len(pics_selectied) > 0:
                    pics_data = []
                    for pic_file in pics_selectied:
                        pic_path = os.path.join(f"{pics_path}/chr{ww}_pics", pic_file)
                        with Image.open(pic_path) as img:
                            # 将图像转换为RGB模式，以防它是其他模式（如RGBA）
                            img = img.convert('RGB')
                            # 将图像数据转换为numpy数组
                            img_array = np.array(img)
                        pics_data.append(img_array)
                    pics_data = np.stack(pics_data, axis=0)
                
                
                x_train_name = datapath+"/" + chr_name1 + '_' + str(start)  +  '_' + str(start + 10000000) + '.npy'
                print(x_train_name)
                x_t = np.load(x_train_name)

            except FileNotFoundError:
                print("File not found:", x_train_name.split('/')[-1])
                start = start + 10000000
                continue
            else:
                index = [area for area in area_list if area*200 >= start and area*200 < start + 10000000]
                if len(x_t) == 0:
                    continue
                data1,data2,picdata1,picdata2 = batchdata(x_t,pics_data,100,0)
                data1 = data1.reshape(-1,100,200,5)
                data1 = np.expand_dims(data1, axis=-1)
                picdata1 = picdata1.reshape(-1,100,200,200,3)
                if len(data1) == 0:
                    predict1 = np.array([])
                if len(data1)!= 0:
                    predict1 = predict_ins.predict(
                        [data1, picdata1],  # 两个输入组成的列表
                        batch_size=pred_bs,
                        verbose=1
                    )
                    
                    # datatmp1 = tf.data.Dataset.from_tensor_slices(data1)
                    # datatmp_pic = tf.data.Dataset.from_tensor_slices(picdata1)  # 确保picdata1存在且形状正确

                    # # 将两种输入组合成元组
                    # dataset = tf.data.Dataset.zip((datatmp1, datatmp_pic))
                    # dataset = dataset.batch(80)

                    # # 正确调用预测方法
                    # predict1 = predict_ins.predict(dataset) 
                if len(data2)!=0:
                    data2 = data2.reshape(1,-1,200,5,1)
                    picdata2 = picdata2.reshape(1,-1,200,200,3)
                    #datatmp2 = tf.data.Dataset.from_tensor_slices(data2)
                    predict2 = predict_ins.predict((data2, picdata2), batch_size=pred_bs)
                else:
                    predict2 = np.array([])
                predict1 = predict1.flatten()
                predict2 = predict2.flatten()
                lis = []
                lis.extend(predict1)
                lis.extend(predict2)
                base = np.array(lis)
                
                data3,data4,picdata3,picdata4 = batchdata(x_t,pics_data,100,100)
                data3 = data3.reshape(-1,100,200,5)
                data3 = np.expand_dims(data3, axis=-1)
                picdata3 = picdata3.reshape(-1,100,200,200,3)
                if len(data3) == 0:
                    predict3 = np.array([])
                if len(data3)!=0:
                    # datatmp3 = tf.data.Dataset.from_tensor_slices(data3).batch(80)
                    # predict3 = predict_ins.predict(datatmp3)
                    predict3 = predict_ins.predict(
                        [data3, picdata3],  # 两个输入组成的列表
                        batch_size=pred_bs,
                        verbose=1
                    )
                if len(data4)!=0:
                    data4 = data4.reshape(1,-1,200,5,1)
                    picdata4 = picdata4.reshape(1,-1,200,200,3)
                    predict4 = predict_ins.predict((data4, picdata4), batch_size=pred_bs)
                else:
                    predict4 = np.array([])
                predict3 = predict3.flatten()
                predict4 = predict4.flatten() 
                lis2 = []
                lis2.extend(predict3)
                lis2.extend(predict4)
                base1 = np.array(lis2)
                base = predcit_step(base,base1)
                

                data5,data6,picdata5,picdata6 = batchdata(x_t,pics_data,100,50)
                data5 = data5.reshape(-1,100,200,5)
                data5 = np.expand_dims(data5, axis=-1)
                picdata5 = picdata5.reshape(-1,100,200,200,3)
                if len(data5) == 0:
                    predict5 = np.array([])
                if len(data5)!=0:
                    # datatmp5 = tf.data.Dataset.from_tensor_slices(data5).batch(80)
                    # predict5 = predict_ins.predict(datatmp5)
                    predict5 = predict_ins.predict(
                        [data5, picdata5],  # 两个输入组成的列表
                        batch_size=pred_bs,
                        verbose=1
                    )
                    
                if len(data6)!=0:
                    data6 = data6.reshape(1,-1,200,5,1)
                    picdata6 = picdata6.reshape(1,-1,200,200,3)
                    predict6 = predict_ins.predict((data6, picdata6), batch_size=pred_bs)
                else:
                    predict6 = np.array([])
                predict5 = predict5.flatten()
                predict6 = predict6.flatten() 
                lis3 = []
                lis3.extend(predict5)
                lis3.extend(predict6)
                base2 = np.array(lis3)
                base = predcit_step(base,base2)
        
                data7,data8,picdata7,picdata8 = batchdata(x_t,pics_data,100,150)
                data7 = data7.reshape(-1,100,200,5)
                data7 = np.expand_dims(data7, axis=-1)
                picdata7 = picdata7.reshape(-1,100,200,200,3)
                if len(data7) == 0:
                    predict7 = np.array([])
                if len(data7)!=0:
                    # datatmp7 = tf.data.Dataset.from_tensor_slices(data7).batch(80)
                    # predict7 = predict_ins.predict(datatmp7)
                    predict7 = predict_ins.predict(
                        [data7, picdata7],  # 两个输入组成的列表
                        batch_size=pred_bs,
                        verbose=1
                    )
                if len(data8)!=0:
                    data8 = data8.reshape(1,-1,200,5,1)
                    picdata8 = picdata8.reshape(1,-1,200,200,3)
                    predict8 = predict_ins.predict((data8, picdata8), batch_size=pred_bs)
                else:
                    predict8 = np.array([])
                predict7 = predict7.flatten()
                predict8 = predict8.flatten() 
                lis4 = []
                lis4.extend(predict7)
                lis4.extend(predict8)
                base3 = np.array(lis4)
                base = predcit_step(base,base3)

                
                
                contig, start = chr_name, start
                resultlist += mergedeleton_long(base, index,contig,bamfile)
                
                #print(start)
                start = start + 10000000
                #history = model.evaluate(x_t[:,:-1].reshape(-1, 200, 6), x_t[:,-1],batch_size = 2048)
                #print(chr_name,i)
    ins = []
    for read in resultlist[1:]:
        if read[3] >= int(support) and read[2] >= 50:
            
            ins.append([read[0],read[1],read[2],read[3],'INS','.'])
            
    tovcf(ins,contig2length,outvcfpath)
        
    return resultlist

# 主程序
if __name__ == "__main__":
    # 环境变量控制：
    #   SKIP_TRAIN=1  -> 跳过训练
    #   RUN_PREDICT=1 -> 执行预测流程
    #   EPOCHS=<int>  -> 训练轮数(默认300)
    #   QUICK_RUN=1   -> 仅快速构建模型(已在上方实现)
    epochs = int(os.environ.get('EPOCHS', '300'))
    do_train = os.environ.get('SKIP_TRAIN', '0') != '1'
    do_predict = os.environ.get('RUN_PREDICT', '0') == '1'
    image_backbone = os.environ.get('IMAGE_BACKBONE', 'efficientnetb0')
    use_pretrained = os.environ.get('USE_PRETRAINED', '1') == '1'

    if do_train:
        try:
            print(f"[INFO] 开始训练: backbone={image_backbone}, epochs={epochs}, pretrained={use_pretrained}")
            train(model_path, pic_path, tensor_path, labels_path, main_path,
                  image_backbone=image_backbone, image_output_dim=128, use_pretrained=use_pretrained,
                  epochs=epochs)
        except Exception as e:
            print("程序崩溃，错误信息：", e)
            import traceback
            traceback.print_exc()
            # 不强制 raise，允许后续预测或退出
    else:
        print("[INFO] 已跳过训练 (SKIP_TRAIN=1)")

    if do_predict:
        contigg = [i for i in range (12,23)]
        try:
            print("[INFO] 开始预测...")
            resultlist = predict_funtion(area_list_path, model_path, tensor_path, pic_path, bamfile_path, outvcfpath, contigg, 10)
            best_resultlist = predict_funtion(area_list_path, best_model_path, tensor_path, pic_path, bamfile_path, best_outvcfpath, contigg, 10)
            print("[INFO] 预测完成。")
        except Exception as e:
            print("[WARN] 预测阶段出错: ", e)
    else:
        print("[INFO] 未执行预测 (设置 RUN_PREDICT=1 可启用)")
