# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time, traceback, json # Добавляем json
from typing import Tuple, List, Optional

# =============================================================================
# 0. Гиперпараметры и Конфигурация
# =============================================================================
# --- Модель ---
VOCAB_SIZE_ESTIMATE = 128000 # Начальная оценка, будет обновлена из токенизатора
D_MODEL = 128
MAX_SEQ_LEN = 1024 # (1024)
SEGMENT_LENGTH = 64 # Должно делить MAX_SEQ_LEN
NUM_CORE_BLOCKS = 1  # Попробуем 2 блока для большей емкости

# --- Обучение ---
DATASET_FILENAME = "dataset.json" # Имя файла с данными
EPOCHS = 10  # Уменьшим количество эпох для начала
BATCH_SIZE_PER_REPLICA = 8 # Можно увеличить, если память TPU/GPU позволяет
LEARNING_RATE = 0.01 # Более стандартный LR для старта

# --- Генерация ---
TEST_PROMPT = "В каком году было установлено в Японии 3 городских префектур фу — Токио, Киото и Осака?" # Пример промпта для генерации
MAX_GEN_LENGTH = 100 # Максимальное количество генерируемых токенов

# --- Токенизатор (Заглушка, если реальный не найден) ---
class PlaceholderTokenizer:
    vocab_size = VOCAB_SIZE_ESTIMATE
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    def encode(self, text): return [ord(c) for c in text[:50] if ord(c) < self.vocab_size]
    def decode(self, tokens): return "".join([chr(t) for t in tokens if t >= 32])

# =============================================================================
# 1. Загрузка Токенизатора
# =============================================================================
try:
    from deepseek_tokenizer import ds_token
    print("✅ Реальный deepseek_tokenizer загружен.")
    VOCAB_SIZE = ds_token.vocab_size # Обновляем размер словаря
except ImportError:
    print("WARNING: deepseek_tokenizer not found. Using placeholder values.")
    ds_token = PlaceholderTokenizer()
    VOCAB_SIZE = ds_token.vocab_size # Используем оценку
    print(f"Using placeholder vocab size: {VOCAB_SIZE}")

# Обновляем токен ID из загруженного токенизатора (или заглушки)
pad_token_id = getattr(ds_token, 'pad_token_id', 0)
bos_token_id = getattr(ds_token, 'bos_token_id', 1)
eos_token_id = getattr(ds_token, 'eos_token_id', 2)
print(f"Токены: PAD={pad_token_id}, BOS={bos_token_id}, EOS={eos_token_id}, Vocab Size: {VOCAB_SIZE}")

# Проверка гиперпараметров
if MAX_SEQ_LEN % SEGMENT_LENGTH != 0:
    raise ValueError(f"MAX_SEQ_LEN ({MAX_SEQ_LEN}) must be divisible by SEGMENT_LENGTH ({SEGMENT_LENGTH})")
if D_MODEL % 2 != 0:
    D_MODEL += 1
    print(f"Adjusted D_MODEL to be even: {D_MODEL}")

# =============================================================================
# 2. TPU/GPU Инициализация
# =============================================================================
strategy = tf.distribute.get_strategy()
tpu_initialized = False
try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    print(f"Попытка подключения к TPU: {tpu_resolver.master()}...")
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print(f"✅ TPU инициализирован, устройств: {strategy.num_replicas_in_sync}")
    tpu_initialized = True
except Exception as e:
    print(f"❌ Не удалось инициализировать TPU: {e}")
    if "Could not find SliceBuilder port" in str(e):
         print("Примечание: Ошибка 'Could not find SliceBuilder port' может быть связана с метриками, но часто не мешает инициализации.")
    print("Пробуем MirroredStrategy...")
    try:
        strategy = tf.distribute.MirroredStrategy()
        print(f"✅ Использована MirroredStrategy, устройств: {strategy.num_replicas_in_sync}")
    except Exception as e_mirror:
        print(f"❌ Не удалось инициализировать MirroredStrategy: {e_mirror}. Использована стратегия по умолчанию.")
        strategy = tf.distribute.get_strategy()
    print(f"Работаем на {strategy.num_replicas_in_sync} устройстве(ах) (CPU/GPU)")

finally:
    tf.get_logger().setLevel('INFO')

# Рассчитываем глобальный батч после определения стратегии
num_replicas = strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * num_replicas
print(f"Global batch size: {GLOBAL_BATCH_SIZE} ({BATCH_SIZE_PER_REPLICA} per replica * {num_replicas} replicas)")

# tf.keras.mixed_precision.set_global_policy('mixed_bfloat16') # Опционально для TPU/Ampere GPU

# =============================================================================
# 3. Функции Предобработки Данных
# =============================================================================
def tokenize_and_pad(texts: List[str], max_len: int) -> np.ndarray:
    """Токенизирует список текстов и дополняет их до max_len."""
    tokenized_batch = []
    for text in texts:
        # Убираем BOS/EOS токены токенизатора, добавим свои явно
        encoded = ds_token.encode(text)
        # Фильтрация токенов больше размера словаря (на всякий случай)
        encoded = [t for t in encoded if t < VOCAB_SIZE]

        # Оставляем место для BOS и EOS
        max_len_adj = max_len - 2
        if len(encoded) > max_len_adj:
            encoded = encoded[:max_len_adj] # Обрезаем, если длиннее

        # Добавляем BOS и EOS
        tokens = [bos_token_id] + encoded + [eos_token_id]

        # Дополняем PAD токенами
        pad_len = max_len - len(tokens)
        if pad_len > 0:
            tokens += [pad_token_id] * pad_len
        elif pad_len < 0: # Если после добавления BOS/EOS стало длиннее max_len
            tokens = tokens[:max_len]
            # Убедимся, что последний токен не PAD, если только вся послед-ть не PAD
            if tokens[-1] == pad_token_id and any(t != pad_token_id for t in tokens[:-1]):
                 # Лучше заменить на EOS, если обрезали EOS
                 tokens[-1] = eos_token_id
            elif tokens[-1] != eos_token_id and tokens[-1] != pad_token_id:
                 # Если последний токен не EOS и не PAD, ставим EOS для консистентности
                 tokens[-1] = eos_token_id


        assert len(tokens) == max_len, f"Final length mismatch: {len(tokens)} vs {max_len} for text: '{text[:50]}...'"
        tokenized_batch.append(tokens)
    return np.array(tokenized_batch, dtype=np.int32)

# =============================================================================
# 4. Определение Слоев Модели (RMSNorm, RoPE, S6, GLCM, MoE, GFCU, CoreBlock, SegmentProcessing, TiedDense)
# (Код слоев остается тем же, что и в предыдущем ответе)
# =============================================================================

# 4.1 RMSNorm
class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self._gamma = None
    def build(self, input_shape):
        if self._gamma is None:
            dim = input_shape[-1]
            if dim is None: raise ValueError("Last dimension cannot be None for RMSNorm")
            self._gamma = self.add_weight(name="gamma", shape=(dim,), initializer=tf.ones_initializer(), trainable=True)
        super().build(input_shape)
    @tf.function
    def call(self, x):
        input_dtype = x.dtype; x_float32 = tf.cast(x, tf.float32)
        rms = tf.math.sqrt(tf.reduce_mean(tf.square(x_float32), axis=-1, keepdims=True) + self.epsilon)
        if self._gamma is None: raise ValueError("RMSNorm gamma not initialized.")
        gamma_float32 = tf.cast(self._gamma, tf.float32)
        normalized = gamma_float32 * (x_float32 / rms)
        return tf.cast(normalized, input_dtype)

# 4.2 RoPE
def get_rope_embeddings(seq_len: int, dim: int) -> Tuple[tf.Tensor, tf.Tensor]:
    if dim % 2 != 0: raise ValueError(f"Dimension {dim} must be even")
    half_dim = dim // 2; position = tf.cast(tf.range(seq_len), tf.float32)
    float_half_dim = tf.cast(half_dim, tf.float32)
    inv_freq = 1.0 / (10000 ** (tf.range(0, half_dim, dtype=tf.float32) / float_half_dim))
    sinusoid_inp = tf.einsum('i,j->ij', position, inv_freq)
    sin = tf.sin(sinusoid_inp); cos = tf.cos(sinusoid_inp)
    return tf.cast(sin, tf.float32), tf.cast(cos, tf.float32)

@tf.function
def apply_rotary(x: tf.Tensor, sin: tf.Tensor, cos: tf.Tensor) -> tf.Tensor:
    input_dtype = x.dtype; x_float = tf.cast(x, tf.float32)
    sin_float = tf.cast(sin, tf.float32); cos_float = tf.cast(cos, tf.float32)
    x_rank = len(tf.shape(x_float)); rope_rank = len(tf.shape(sin_float))
    if x_rank == 3 and rope_rank == 2: sin_float = tf.expand_dims(sin_float, 0); cos_float = tf.expand_dims(cos_float, 0)
    elif x_rank == 2 and rope_rank == 2: sin_float = sin_float[0]; cos_float = cos_float[0]; sin_float = tf.expand_dims(sin_float, 0); cos_float = tf.expand_dims(cos_float, 0)
    x1 = x_float[..., ::2]; x2 = x_float[..., 1::2]
    rotated_x1 = x1 * cos_float - x2 * sin_float; rotated_x2 = x1 * sin_float + x2 * cos_float
    x_shape = tf.shape(x_float); output_zeros = tf.zeros_like(x_float)
    if x_rank == 3:
        B, T, D = x_shape[0], x_shape[1], x_shape[2]; half_dim = D // 2
        even_indices_d = tf.range(0, D, 2, dtype=tf.int32); odd_indices_d = tf.range(1, D, 2, dtype=tf.int32)
        b_indices = tf.tile(tf.reshape(tf.range(B), [B, 1, 1]), [1, T, half_dim]); t_indices = tf.tile(tf.reshape(tf.range(T), [1, T, 1]), [B, 1, half_dim])
        even_d_indices = tf.tile(tf.reshape(even_indices_d, [1, 1, half_dim]), [B, T, 1]); odd_d_indices = tf.tile(tf.reshape(odd_indices_d, [1, 1, half_dim]), [B, T, 1])
        indices_even = tf.stack([b_indices, t_indices, even_d_indices], axis=-1); indices_odd = tf.stack([b_indices, t_indices, odd_d_indices], axis=-1)
    elif x_rank == 2:
        B, D = x_shape[0], x_shape[1]; half_dim = D // 2
        even_indices_d = tf.range(0, D, 2, dtype=tf.int32); odd_indices_d = tf.range(1, D, 2, dtype=tf.int32)
        b_indices = tf.tile(tf.reshape(tf.range(B), [B, 1]), [1, half_dim])
        even_d_indices = tf.tile(tf.reshape(even_indices_d, [1, half_dim]), [B, 1]); odd_d_indices = tf.tile(tf.reshape(odd_indices_d, [1, half_dim]), [B, 1])
        indices_even = tf.stack([b_indices, even_d_indices], axis=-1); indices_odd = tf.stack([b_indices, odd_d_indices], axis=-1)
    else: raise ValueError(f"Unsupported input rank for apply_rotary: {x_rank}")
    output_updated_even = tf.tensor_scatter_nd_update(output_zeros, indices_even, rotated_x1)
    output_final = tf.tensor_scatter_nd_update(output_updated_even, indices_odd, rotated_x2)
    return tf.cast(output_final, input_dtype)

# 4.3 S6Layer
class S6Layer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, d_state: Optional[int] = None, N: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.d_model_input_dim = d_model; self.d_model = d_model
        self.d_state = d_state if d_state is not None else d_model; self.N = N
        self.d_inner = 2 * d_model; self.proj_inner = tf.keras.layers.Dense(self.d_inner, name="proj_inner")
        self.dense_delta = tf.keras.layers.Dense(self.d_state, activation=tf.nn.softplus, name="dense_delta")
        self.dense_B = tf.keras.layers.Dense(self.d_state * self.N, name="dense_B")
        self.dense_C = tf.keras.layers.Dense(self.d_state * self.N, name="dense_C")
        self.proj_out = tf.keras.layers.Dense(self.d_model, name="proj_out"); self.rmsnorm = RMSNorm(name="rmsnorm_s6_out"); self.A_log = None
    def build(self, input_shape):
        if self.A_log is None: A_init_val = np.log(np.arange(1, self.N + 1, dtype=np.float32)).reshape(1, self.N); A_init_val = np.tile(A_init_val, (self.d_state, 1)); self.A_log = self.add_weight(name="A_log", shape=(self.d_state, self.N), initializer=tf.constant_initializer(A_init_val), trainable=True)
        self.proj_inner.build(input_shape); inner_shape = input_shape[:-1] + (self.d_inner,)
        self.dense_delta.build(inner_shape); self.dense_B.build(inner_shape); self.dense_C.build(inner_shape)
        state_output_shape = tf.TensorShape([None, self.d_state]); model_output_shape_for_norm = tf.TensorShape([None, self.d_model])
        if not self.proj_out.built: self.proj_out.build(state_output_shape)
        self.rmsnorm.build(model_output_shape_for_norm); super().build(input_shape)
    def step_fn(self, h_prev: tf.Tensor, x_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        B = tf.shape(x_t)[0]; x_t_float = tf.cast(x_t, dtype=tf.float32); h_prev_float = tf.cast(h_prev, dtype=tf.float32)
        if self.A_log is None: raise ValueError("S6Layer.A_log is not initialized.")
        A_neg_exp_float = -tf.exp(tf.cast(self.A_log, dtype=tf.float32)); x_proj = self.proj_inner(x_t_float); delta = self.dense_delta(x_proj)
        B_proj = tf.reshape(self.dense_B(x_proj), (B, self.d_state, self.N)); C_proj = tf.reshape(self.dense_C(x_proj), (B, self.d_state, self.N))
        delta_A = tf.exp(tf.einsum('bd,dn->bdn', delta, A_neg_exp_float)); delta_B = tf.einsum('bd,bdn->bdn', delta, B_proj)
        delta_A_minus_1 = delta_A - 1.0; delta_times_A = tf.einsum('bd,dn->bdn', delta, A_neg_exp_float); denom = delta_times_A
        v = tf.math.divide_no_nan(delta_A_minus_1, denom); v = tf.where(tf.math.is_finite(v), v, tf.ones_like(v)); B_bar = v * delta_B
        h_t = delta_A * h_prev_float + B_bar; y_t_inner = tf.reduce_sum(h_t * C_proj, axis=-1); y_t_inner.set_shape([None, self.d_state])
        y_t = self.proj_out(y_t_inner); y_t_norm = self.rmsnorm(y_t); return h_t, y_t_norm
    def call(self, x: tf.Tensor, initial_state: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        B = tf.shape(x)[0]; input_dtype = x.dtype
        if initial_state is None: h0 = tf.zeros((B, self.d_state, self.N), dtype=tf.float32)
        else: h0 = tf.cast(initial_state, dtype=tf.float32); tf.debugging.assert_shapes([(h0, [None, self.d_state, self.N])])
        init_y = tf.zeros((B, self.proj_out.units), dtype=tf.float32); init_tuple = (h0, init_y); x_tm = tf.transpose(x, perm=[1, 0, 2])
        x_tm_float = tf.cast(x_tm, dtype=tf.float32); scanned_outputs = tf.scan(lambda prev, x_curr: self.step_fn(prev[0], x_curr), elems=x_tm_float, initializer=init_tuple)
        h_all, y_all = scanned_outputs; h_all_transposed = tf.transpose(h_all, perm=[1, 0, 2, 3]); y_all_transposed = tf.transpose(y_all, perm=[1, 0, 2])
        y_all_casted = tf.cast(y_all_transposed, dtype=input_dtype); last_state = h_all_transposed[:, -1, :, :]; return y_all_casted, last_state
    def step(self, x_t: tf.Tensor, h_prev: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.built: x_t.set_shape([None, self.d_model_input_dim]); h_prev.set_shape([None, self.d_state, self.N])
        x_t_float = tf.cast(x_t, tf.float32); h_prev_float = tf.cast(h_prev, tf.float32); h_next, y_next = self.step_fn(h_prev_float, x_t_float); return h_next, y_next

# 4.4 GLCMLayer
class GLCMLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        if d_model % 2 != 0: raise ValueError(f"d_model ({d_model}) must be even")
        self.d_model = d_model; self.kernel_size = kernel_size; self.proj = tf.keras.layers.Dense(d_model * 2, name="proj_glcm")
        self.depthwise_conv = tf.keras.layers.DepthwiseConv1D(kernel_size, padding='same', name="dw_conv"); self.activation = tf.keras.layers.Activation('swish'); self.rmsnorm = RMSNorm(name="rmsnorm_glcm")
    def build(self, input_shape):
        self.proj.build(input_shape); conv_input_shape = input_shape[:-1] + (self.d_model,)
        self.depthwise_conv.build(conv_input_shape); self.rmsnorm.build(conv_input_shape); super().build(input_shape)
    @tf.function
    def call(self, x: tf.Tensor, sin: tf.Tensor, cos: tf.Tensor) -> tf.Tensor:
        input_dtype = x.dtype; x_float = tf.cast(x, tf.float32); sin_float = tf.cast(sin, tf.float32); cos_float = tf.cast(cos, tf.float32)
        proj_out = self.proj(x_float); x_a, x_b = proj_out[..., :self.d_model], proj_out[..., self.d_model:]
        x_a_rotated = apply_rotary(x_a, sin_float, cos_float); conv_out = self.depthwise_conv(x_a_rotated)
        conv_out_activated = self.activation(conv_out); gate = tf.nn.sigmoid(tf.cast(x_b, tf.float32)); gated_output = conv_out_activated * gate
        normalized_output = self.rmsnorm(gated_output); return tf.cast(normalized_output, input_dtype)

# 4.5 MoELayer
class MoELayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_experts=16, k=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_experts, self.k = d_model, num_experts, k
        self.experts = [tf.keras.Sequential([tf.keras.layers.Dense(d_model * 4, activation='swish', name=f"expert_{i}_dense1"), tf.keras.layers.Dense(d_model, name=f"expert_{i}_dense2")], name=f"expert_{i}") for i in range(num_experts)]
        self.gate_dense = tf.keras.layers.Dense(num_experts, name="gate_dense", use_bias=False)
    def build(self, input_shape):
        self.gate_dense.build(input_shape); [expert.build(input_shape) for expert in self.experts]; super().build(input_shape)
    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        input_dtype = x.dtype; x_float = tf.cast(x, tf.float32); B, T = tf.shape(x_float)[0], tf.shape(x_float)[1]; T = T if T is not None else tf.shape(x_float)[1]
        gate_logits = self.gate_dense(x_float); actual_k = tf.minimum(self.k, self.num_experts); topk = tf.math.top_k(gate_logits, k=actual_k); weights = tf.nn.softmax(topk.values, axis=-1)
        idx_b = tf.range(B); idx_t = tf.range(T); idx_b, idx_t = tf.meshgrid(idx_b, idx_t, indexing='ij')
        idx_b = tf.tile(tf.expand_dims(idx_b, axis=-1), [1, 1, actual_k]); idx_t = tf.tile(tf.expand_dims(idx_t, axis=-1), [1, 1, actual_k])
        full_indices = tf.stack([idx_b, idx_t, topk.indices], axis=-1)
        expert_outputs_list = [expert(x_float) for expert in self.experts]; stacked_expert_outputs = tf.stack(expert_outputs_list, axis=2)
        topk_expert_outputs = tf.gather_nd(stacked_expert_outputs, full_indices); weights_expanded = tf.expand_dims(weights, axis=-1)
        weighted_expert_outputs = topk_expert_outputs * weights_expanded; final_output_float = tf.reduce_sum(weighted_expert_outputs, axis=2)
        return tf.cast(final_output_float, input_dtype)

# 4.6 GFCULayer
class GFCULayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs); self.d_model = d_model; self.gate_dense1 = tf.keras.layers.Dense(d_model, activation='sigmoid', name="gate1"); self.gate_dense2 = tf.keras.layers.Dense(d_model, activation='sigmoid', name="gate2"); self.rmsnorm = RMSNorm(name="rmsnorm_gfcu")
    def build(self, input_shape): self.gate_dense1.build(input_shape); self.gate_dense2.build(input_shape); self.rmsnorm.build(input_shape); super().build(input_shape)
    @tf.function
    def call(self, s6_out: tf.Tensor, glcm_out: tf.Tensor, x: tf.Tensor):
        input_dtype = x.dtype; x_float = tf.cast(x, tf.float32); gate1 = self.gate_dense1(x_float); gate2 = self.gate_dense2(x_float)
        s6_out_float = tf.cast(s6_out, tf.float32); glcm_out_float = tf.cast(glcm_out, tf.float32); fused_float = gate1 * s6_out_float + gate2 * glcm_out_float
        fused_norm_float = self.rmsnorm(fused_float); return tf.cast(fused_norm_float, input_dtype)

# 4.7 AetheriusCoreBlock
class AetheriusCoreBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_moe_experts=16, moe_k=2, **kwargs): # Добавим параметры MoE
        super().__init__(**kwargs)
        if d_model % 2 != 0: raise ValueError(f"d_model ({d_model}) must be even")
        self.d_model = d_model
        self.s6 = S6Layer(d_model, name="s6_layer")
        self.glcm = GLCMLayer(d_model, name="glcm_layer")
        self.moe = MoELayer(d_model, num_experts=num_moe_experts, k=moe_k, name="moe_layer") # Используем параметры
        self.gfcu = GFCULayer(d_model, name="gfcu_layer")
        self.rmsnorm_final = RMSNorm(name="rmsnorm_acb_final")
    def build(self, input_shape): [layer.build(input_shape) for layer in [self.s6, self.glcm, self.moe, self.gfcu, self.rmsnorm_final]]; super().build(input_shape)
    def call(self, x: tf.Tensor, sin: tf.Tensor, cos: tf.Tensor, h_prev: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        input_dtype = x.dtype; x_float = tf.cast(x, tf.float32); residual = x_float
        s6_out_y, h_new = self.s6(x, initial_state=h_prev); s6_out_y_float = tf.cast(s6_out_y, tf.float32)
        glcm_out = self.glcm(x, sin=sin, cos=cos); glcm_out_float = tf.cast(glcm_out, tf.float32)
        fused = self.gfcu(s6_out_y_float, glcm_out_float, x_float); fused_float = tf.cast(fused, tf.float32)
        moe_out = self.moe(fused_float); moe_out_float = tf.cast(moe_out, tf.float32)
        summed = moe_out_float + residual; output_float = self.rmsnorm_final(summed)
        return tf.cast(output_float, input_dtype), h_new
    def step(self, x_t: tf.Tensor, h_prev_step: tf.Tensor, sin_step: tf.Tensor, cos_step: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.built: x_t.set_shape([None, self.d_model]); h_prev_step.set_shape([None, self.s6.d_state, self.s6.N]); sin_step.set_shape([1, 1, self.d_model // 2]); cos_step.set_shape([1, 1, self.d_model // 2])
        x_t_float = tf.cast(x_t, tf.float32); h_prev_float = tf.cast(h_prev_step, tf.float32); residual = x_t_float
        s6_state_h, s6_output_y = self.s6.step(x_t_float, h_prev_float)
        x_t_expanded = tf.expand_dims(x_t_float, axis=1); glcm_out_casted = self.glcm(x_t_expanded, sin=sin_step, cos=cos_step); glcm_out_squeezed = tf.squeeze(glcm_out_casted, axis=1); glcm_out_float = tf.cast(glcm_out_squeezed, tf.float32)
        fused_casted = self.gfcu(s6_output_y, glcm_out_float, x_t_float); fused_float = tf.cast(fused_casted, tf.float32)
        moe_input = tf.expand_dims(fused_float, axis=1); moe_input.set_shape([None, 1, self.d_model]); moe_out_casted = self.moe(moe_input); moe_out_squeezed = tf.squeeze(moe_out_casted, axis=1); moe_out_float = tf.cast(moe_out_squeezed, tf.float32)
        summed = moe_out_float + residual; output = self.rmsnorm_final(summed); output_float = tf.cast(output, tf.float32)
        return output_float, s6_state_h

# 4.8 SegmentProcessingLayer
class SegmentProcessingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, segment_length: int, max_seq_len: int, core_blocks: List[AetheriusCoreBlock], **kwargs):
        super().__init__(**kwargs)
        if d_model % 2 != 0 or max_seq_len % segment_length != 0: raise ValueError("Invalid dims for SegmentProcessingLayer")
        self.d_model, self.segment_length, self.max_seq_len = d_model, segment_length, max_seq_len
        self.num_segments = max_seq_len // segment_length; self.core_blocks = core_blocks
        sin_seg, cos_seg = get_rope_embeddings(self.segment_length, self.d_model); self.sin_segment = tf.expand_dims(sin_seg, 0); self.cos_segment = tf.expand_dims(cos_seg, 0)
    def build(self, input_shape): dummy_segment_shape = tf.TensorShape([None, self.segment_length, input_shape[-1]]); [block.build(dummy_segment_shape) for block in self.core_blocks if not block.built]; super().build(input_shape)
    @tf.function
    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...]]:
        B, T, D = tf.shape(x)[0], tf.shape(x)[1], x.shape[-1]; D = D if D is not None else tf.shape(x)[-1]; input_dtype = x.dtype
        tf.debugging.assert_equal(T, self.max_seq_len, message=f"Input len {T} != max_seq_len {self.max_seq_len}")
        segments = tf.reshape(x, [B, self.num_segments, self.segment_length, D]); segments_time_major = tf.transpose(segments, perm=[1, 0, 2, 3])
        def seg_step(prev_states_tuple, current_segment):
            segment_input = current_segment; next_states_list = []
            for i in range(len(self.core_blocks)): block = self.core_blocks[i]; h_prev = prev_states_tuple[i]; segment_output, h_next = block(segment_input, sin=self.sin_segment, cos=self.cos_segment, h_prev=h_prev); next_states_list.append(h_next); segment_input = segment_output
            return tuple(next_states_list), segment_output
        initial_states = []
        for block in self.core_blocks:
             if not block.s6.built: s6_input_shape = tf.TensorShape([None, self.segment_length, D]); block.s6.build(s6_input_shape); print(f"Built S6 in {block.name} for state shape")
             state_shape = [B, block.s6.d_state, block.s6.N]; initial_states.append(tf.zeros(state_shape, dtype=tf.float32))
        initial_states_tuple = tuple(initial_states); initial_y_segment = tf.zeros([B, self.segment_length, D], dtype=input_dtype); initializer_tuple = (initial_states_tuple, initial_y_segment)
        scanned_outputs = tf.scan(lambda prev, seg: seg_step(prev[0], seg), elems=segments_time_major, initializer=initializer_tuple)
        final_states_sequences_tuple, output_segments_sequence = scanned_outputs; final_states_tuple = tuple(s[-1] for s in final_states_sequences_tuple)
        output_segments_list_transposed = tf.transpose(output_segments_sequence, perm=[1, 0, 2, 3]); full_output = tf.reshape(output_segments_list_transposed, [B, self.max_seq_len, D])
        return full_output, final_states_tuple

# 4.9 TiedDense
class TiedDense(tf.keras.layers.Layer):
    def __init__(self, embed_layer: tf.keras.layers.Embedding, **kwargs):
        super().__init__(**kwargs); self.embed_layer = embed_layer; assert isinstance(embed_layer, tf.keras.layers.Embedding)
    def build(self, input_shape):
        if not self.embed_layer.built: self.embed_layer.build(tf.TensorShape([None]))
        super().build(input_shape)
    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        embedding_kernel = self.embed_layer.embeddings; inputs_float = tf.cast(inputs, tf.float32); embedding_kernel_float = tf.cast(embedding_kernel, tf.float32)
        logits = tf.matmul(inputs_float, embedding_kernel_float, transpose_b=True); return logits

# =============================================================================
# 5. Определение Модели Aetherius
# =============================================================================
class AetheriusModel(tf.keras.Model):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, segment_length: int, core_blocks: List[AetheriusCoreBlock], **kwargs):
        super().__init__(name="aetherius_model", **kwargs)
        self.vocab_size, self.d_model, self.max_seq_len, self.segment_length = vocab_size, d_model, max_seq_len, segment_length
        if max_seq_len % segment_length != 0: raise ValueError("max_seq_len must be divisible by segment_length")
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model, name="token_embedding")
        self.core_blocks = core_blocks
        self.segment_processor = SegmentProcessingLayer(d_model, segment_length, max_seq_len, self.core_blocks, name="segment_processor")
        self.final_rmsnorm = RMSNorm(name="final_rmsnorm")
        self.final_dense = TiedDense(self.token_embedding, name="final_dense_tied")
        self._step_compiled = False

    def build(self, input_shape):
        self.token_embedding.build(input_shape); embed_output_shape = tf.TensorShape([None, None, self.d_model]); segment_input_shape = tf.TensorShape([None, self.max_seq_len, self.d_model])
        if not self.segment_processor.built: self.segment_processor.build(segment_input_shape)
        if not self.final_rmsnorm.built: self.final_rmsnorm.build(embed_output_shape)
        if not self.final_dense.built: self.final_dense.build(embed_output_shape)
        super().build(input_shape)

    @tf.function
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        compute_dtype = tf.float32; x = self.token_embedding(inputs); x_float = tf.cast(x, compute_dtype); current_seq_len = tf.shape(x_float)[1]
        padding_needed = self.max_seq_len - current_seq_len; x_padded = tf.cond(padding_needed > 0, lambda: tf.pad(x_float, [[0, 0], [0, padding_needed], [0, 0]]), lambda: x_float)
        x_padded.set_shape([None, self.max_seq_len, self.d_model]); processed_output, _ = self.segment_processor(x_padded); processed_output_float = tf.cast(processed_output, compute_dtype)
        output_norm = self.final_rmsnorm(processed_output_float); logits_padded = self.final_dense(output_norm)
        final_logits = tf.cond(padding_needed > 0, lambda: tf.slice(logits_padded, [0, 0, 0], [-1, current_seq_len, -1]), lambda: logits_padded); return final_logits

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def encode_prompt(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        compute_dtype = tf.float32; x = self.token_embedding(inputs); x_float = tf.cast(x, compute_dtype); current_seq_len = tf.shape(x_float)[1]
        padding_needed = self.max_seq_len - current_seq_len; x_padded = tf.cond(padding_needed > 0, lambda: tf.pad(x_float, [[0, 0], [0, padding_needed], [0, 0]]), lambda: x_float)
        x_padded.set_shape([None, self.max_seq_len, self.d_model]); _, final_states_tuple = self.segment_processor(x_padded); return final_states_tuple

    def _get_step_input_signature(self) -> List[tf.TensorSpec]:
        if not self.built: raise RuntimeError("Model must be built before creating step signature.")
        if not self.core_blocks: raise AttributeError("Model 'core_blocks' is empty.")
        first_block = self.core_blocks[0]
        if not first_block.s6.built:
             try: s6_step_input_shape = tf.TensorShape([None, self.d_model]); first_block.s6.build(s6_step_input_shape); print("Warning: Had to build S6 explicitly for step signature.")
             except Exception as e: raise RuntimeError(f"Could not determine S6 d_state/N for step signature: {e}.")
        d_state = first_block.s6.d_state; N = first_block.s6.N
        if d_state is None or N is None: raise ValueError("S6 d_state or N is None after build.")
        s6_state_shape = [None, d_state, N]; s6_state_spec = tf.TensorSpec(shape=s6_state_shape, dtype=tf.float32, name="s6_state")
        state_signature = tuple(s6_state_spec for _ in self.core_blocks); return [tf.TensorSpec(shape=[None], dtype=tf.int32, name="last_token_ids"), state_signature]

    def step(self, last_token_ids: tf.Tensor, previous_states: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...]]:
        sin_step, cos_step = get_rope_embeddings(1, self.d_model); sin_step_exp = tf.expand_dims(sin_step, 0); cos_step_exp = tf.expand_dims(cos_step, 0)
        last_token_expanded = tf.expand_dims(last_token_ids, axis=1); x_t_emb = self.token_embedding(last_token_expanded); x_t = tf.squeeze(x_t_emb, axis=1); x_t_float = tf.cast(x_t, tf.float32)
        current_input = x_t_float; next_states_list = []
        tf.assert_equal(len(previous_states), len(self.core_blocks), message="Num states != num blocks.")
        for i in range(len(self.core_blocks)): block = self.core_blocks[i]; h_prev = previous_states[i]; current_output, h_next = block.step(current_input, h_prev, sin_step_exp, cos_step_exp); next_states_list.append(h_next); current_input = current_output
        final_output_norm_casted = self.final_rmsnorm(current_input); final_output_norm = tf.cast(final_output_norm_casted, tf.float32); logits = self.final_dense(final_output_norm); return logits, tuple(next_states_list)

    def compile_step(self):
       if not self._step_compiled:
           try: step_signature = self._get_step_input_signature(); self.step = tf.function(self.step, input_signature=step_signature); _ = self.step.get_concrete_function(); self._step_compiled = True; print("✅ AetheriusModel.step compiled successfully.")
           except Exception as e: print(f"❌ Failed to compile AetheriusModel.step:"); traceback.print_exc(); print("Warning: step running in Eager mode.")


# =============================================================================
# 6. Загрузка и Подготовка Данных
# =============================================================================
print(f"\nLoading data from {DATASET_FILENAME}...")
input_texts = []
try:
    with open(DATASET_FILENAME, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
             for item in data:
                 if isinstance(item, dict) and 'q' in item and 'a' in item:
                     # Формируем строку в формате "Вопрос [SEP] Ответ" или как вам удобнее
                     # Важно, чтобы модель научилась генерировать ответ после вопроса
                     input_texts.append(f"Q: {item['q']} A: {item['a']}")
                 else:
                     print(f"Warning: Skipping invalid item in dataset: {item}")
        else:
            print(f"Warning: Expected a list in {DATASET_FILENAME}, got {type(data)}. No data loaded.")
    print(f"✅ Loaded {len(input_texts)} samples from {DATASET_FILENAME}.")

except FileNotFoundError:
    print(f"❌ Error: Dataset file '{DATASET_FILENAME}' not found.")
    input_texts = [] # Оставляем пустым, чтобы дальше не упало
except json.JSONDecodeError:
    print(f"❌ Error: Could not decode JSON from '{DATASET_FILENAME}'. Check file format.")
    input_texts = []
except Exception as e:
    print(f"❌ An unexpected error occurred while loading data: {e}")
    input_texts = []

if not input_texts:
    raise SystemExit("No training data loaded. Exiting.")

print("Tokenizing and padding data...")
start_tokenize_time = time.time()
input_data = tokenize_and_pad(input_texts, MAX_SEQ_LEN)
print(f"Tokenization finished in {time.time() - start_tokenize_time:.2f} seconds.")

# Создаем целевые данные (сдвинутые на один токен)
target_data_shift = np.roll(input_data, shift=-1, axis=1)
target_data_shift[:, -1] = pad_token_id # Последний токен не имеет цели
# Маскируем цели там, где исходные данные были PAD
target_data_shift[input_data == pad_token_id] = pad_token_id

print(f"Original number of samples: {len(input_data)}")

# Обработка батчей для распределенной стратегии
num_samples = len(input_data)
num_full_batches = num_samples // GLOBAL_BATCH_SIZE
num_samples_to_use = num_full_batches * GLOBAL_BATCH_SIZE

if num_samples_to_use == 0 and num_samples > 0:
    # Если данных меньше, чем глобальный батч, размножаем их (не идеально, но позволяет запустить)
    repeats = (GLOBAL_BATCH_SIZE + num_samples - 1) // num_samples
    input_data_final = np.tile(input_data, (repeats, 1))[:GLOBAL_BATCH_SIZE]
    target_data_final = np.tile(target_data_shift, (repeats, 1))[:GLOBAL_BATCH_SIZE]
    print(f"Warning: Data tiled {repeats} times ({num_samples} -> {len(input_data_final)}) to fit global batch size {GLOBAL_BATCH_SIZE}.")
elif num_samples_to_use < num_samples:
    # Отбрасываем "хвост", не помещающийся в полный батч
    input_data_final = input_data[:num_samples_to_use]
    target_data_final = target_data_shift[:num_samples_to_use]
    print(f"Using {num_samples_to_use} samples (discarded {num_samples - num_samples_to_use}) to form full batches.")
else:
    # Данных хватает на целое число батчей
    input_data_final = input_data
    target_data_final = target_data_shift
    print(f"Using all {num_samples} samples.")

print(f"Final dataset shape: Input {input_data_final.shape}, Target {target_data_final.shape}")

# Создание tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((input_data_final, target_data_final))
dataset = dataset.shuffle(len(input_data_final)).batch(GLOBAL_BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# =============================================================================
# 7. Создание и Компиляция Модели
# =============================================================================
model = None
with strategy.scope():
    print("\nCreating model instance...")
    core_blocks_list = [AetheriusCoreBlock(D_MODEL, name=f"core_block_{i}") for i in range(NUM_CORE_BLOCKS)]
    model = AetheriusModel(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN, SEGMENT_LENGTH, core_blocks_list)

    print("Building model...")
    sample_input_shape_build = (None, MAX_SEQ_LEN) # Используем MAX_SEQ_LEN для build
    try:
        model.build(input_shape=sample_input_shape_build)
        print("Model built successfully.")
        # Вызов call для компиляции графа обучения
        # Используем BATCH_SIZE_PER_REPLICA, т.к. strategy разделит глобальный батч
        concrete_input = tf.zeros((BATCH_SIZE_PER_REPLICA, MAX_SEQ_LEN), dtype=tf.int32)
        _ = model(concrete_input, training=False)
        print("Model call graph created successfully.")
        # Компиляция step-функции
        model.compile_step()
    except Exception as e:
        print("\n--- Error during explicit model build or call/step compilation ---")
        traceback.print_exc()
        raise SystemExit("Model build/call/step compilation failed.")

    print("Distributing dataset...")
    distributed_dataset = strategy.experimental_distribute_dataset(dataset)
    print("Dataset distributed.")
    try:
        steps_per_epoch = len(distributed_dataset)
    except TypeError: # Может возникнуть для некоторых типов датасетов
        steps_per_epoch = len(input_data_final) // GLOBAL_BATCH_SIZE
    if steps_per_epoch == 0:
         raise ValueError(f"Steps per epoch is 0. Check dataset size ({len(input_data_final)}) and GLOBAL_BATCH_SIZE ({GLOBAL_BATCH_SIZE}).")
    print(f"Steps per epoch: {steps_per_epoch}")

    print("Compiling model for training...")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01, clipnorm=1.0)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(real, pred):
        pred_float32 = tf.cast(pred, tf.float32)
        per_example_loss = loss_object(real, pred_float32)
        mask = tf.math.logical_not(tf.math.equal(real, pad_token_id))
        mask = tf.cast(mask, dtype=per_example_loss.dtype)
        # Используем compute_average_loss для корректного усреднения с маской и по глобальному батчу
        average_loss = tf.nn.compute_average_loss(per_example_loss, sample_weight=mask, global_batch_size=GLOBAL_BATCH_SIZE)
        return average_loss

    model.compile(
        optimizer=optimizer,
        loss=compute_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )
    print("✅ Model compiled for training (with accuracy).")

# =============================================================================
# 8. Обучение Модели
# =============================================================================
print(f"\nStarting training for {EPOCHS} epochs...")
start_time = time.time()
history = None
try:
    if steps_per_epoch > 0 and model is not None and model.optimizer is not None:
        history = model.fit(distributed_dataset, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, verbose=1)
    elif model is None or model.optimizer is None: print("WARNING: Model/optimizer not initialized, skipping training.")
    else: print("WARNING: No steps per epoch, skipping training.")
except Exception as e:
    print("\n--- Error during training ---"); traceback.print_exc()
finally:
    end_time = time.time()
    print("-" * 50); print(f"Training finished in {end_time - start_time:.2f} seconds.")
    if history and hasattr(history, 'history'):
        final_loss = history.history.get('loss', [None])[-1]
        final_accuracy = history.history.get('accuracy', [None])[-1]
        if final_loss is not None: print(f"Final Loss: {final_loss:.4f}")
        if final_accuracy is not None: print(f"Final Accuracy: {final_accuracy:.4f}")
        if final_loss is None and final_accuracy is None: print("Loss/Accuracy history unavailable.")
    else: print("Loss/Accuracy history unavailable.")

# =============================================================================
# 9. Авторегрессионная Генерация
# =============================================================================
def autoregressive_generate_step(model: AetheriusModel, prompt: str, max_gen_len: int, max_seq_len_gen: int) -> str:
    if not model.built: print("Error: Model not built."); return "[Gen Error: Model not built]"
    mode_str = "compiled" if model._step_compiled else "eager (slow)"
    print(f"\nStarting generation (using {mode_str} model.step)...")
    start_gen_time = time.time(); steps_taken = 0; decoded_text = "[Gen Error]"
    try:
        prompt_tokens_unpadded = ds_token.encode(prompt)
        prompt_tokens_unpadded = [t for t in prompt_tokens_unpadded if t < VOCAB_SIZE]
        prompt_tokens = [bos_token_id] + prompt_tokens_unpadded
        if len(prompt_tokens) >= max_seq_len_gen:
            print(f"Warning: Prompt too long ({len(prompt_tokens)}), truncating to {max_seq_len_gen - 1}.")
            prompt_tokens = prompt_tokens[:max_seq_len_gen-1]
        prompt_tf = tf.constant([prompt_tokens], dtype=tf.int32)

        current_states = model.encode_prompt(prompt_tf)
        generated_tokens = list(prompt_tokens)
        last_token_id = generated_tokens[-1]
        print(f"Prompt tokens ({len(generated_tokens)}): {generated_tokens[:50]}..."); print(f"Initial state shapes: {[s.shape for s in current_states]}")
        start_index_of_generation = len(generated_tokens)

        for i in range(max_gen_len):
            steps_taken = i + 1; last_token_tf = tf.constant([last_token_id], dtype=tf.int32)
            try: logits_step, current_states = model.step(last_token_tf, current_states)
            except Exception as step_e: is_compiled_str = "compiled" if model._step_compiled else "eager"; print(f"\n--- Error in {is_compiled_str} model.step (iter {i+1}) ---"); raise step_e
            next_token_id_tf = tf.argmax(logits_step, axis=-1, output_type=tf.int32)[0]
            next_token_id = int(next_token_id_tf.numpy())
            generated_tokens.append(next_token_id); last_token_id = next_token_id
            if next_token_id == eos_token_id: print(f"EOS token found at step {i+1}."); break
        else: print(f"Max length {max_gen_len} reached.")

        model_generated_tokens = generated_tokens[start_index_of_generation:]
        if model_generated_tokens and model_generated_tokens[-1] == eos_token_id: model_generated_tokens = model_generated_tokens[:-1]
        model_generated_tokens = [t for t in model_generated_tokens if t != pad_token_id]
        decoded_text = ds_token.decode(model_generated_tokens)

    except Exception as e: print(f"\n--- Error in generation ---\n{type(e).__name__}: {e}"); traceback.print_exc()
    finally: gen_time = time.time() - start_gen_time; print(f"Generation ({mode_str} step) finished in {gen_time:.2f} seconds ({steps_taken} steps)."); return decoded_text

# Запуск генерации
try:
    if model is not None and model.optimizer is not None:
        print("-" * 50)
        predicted_text = autoregressive_generate_step(model, TEST_PROMPT, MAX_GEN_LENGTH, MAX_SEQ_LEN) # Используем гиперпараметры
        print(f"\nPrompt: {TEST_PROMPT}")
        print(f"Generated text: {predicted_text}")
    else: print("Model was not created/compiled for training, skipping generation.")
except Exception as e: print("\n--- Error during generation call ---"); traceback.print_exc()

# =============================================================================
# 10. Вывод Summary Модели
# =============================================================================
print("\n" + "-" * 50); print("Model Summary:")
try:
    if model is not None and model.built: model.summary(expand_nested=True, show_trainable=True)
    elif model is not None: print("Warning: Model not built, attempting build for summary..."); model.build((None, MAX_SEQ_LEN)); model.summary(expand_nested=True, show_trainable=True)
    else: print("Model instance not available.")
except Exception as e: print(f"\nCould not print model summary: {e}"); traceback.print_exc()

print("\nScript finished.")
