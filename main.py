import matplotlib.pyplot as plt
import cupy as cp
import logging
from data_processor import augment_batch
from data_loader import load_and_preprocess_data
import numpy as np

logger = logging.getLogger("ModelTrainer")

# ===== 1. 分批处理大型数据集 =====


# def train_model_optimized(model, x_train, t_train, x_test, t_test, epochs=10, batch_size=32, learning_rate=0.001):
#     """优化内存使用的训练函数"""
#     # 确保所有数据在GPU上，但采用较小批次
#     optimizer = SGD(learning_rate)
#     train_loss_list = []
#     train_acc_list = []
#     test_acc_list = []
#     best_acc = 0

#     # 计算并记录总批次数，避免重复计算
#     n_batches = (x_train.shape[0] + batch_size - 1) // batch_size

#     for epoch in range(epochs):
#         logger.info(f"Epoch {epoch + 1}/{epochs}")

#         # 每个epoch清空临时变量以释放内存
#         epoch_losses = []
#         cp.get_default_memory_pool().free_all_blocks()

#         # 分批处理数据以减少内存使用
#         batch_indices = cp.random.permutation(x_train.shape[0])

#         for i in range(0, x_train.shape[0], batch_size):
#             batch_idx = batch_indices[i:i+batch_size]

#             # 如果 x_train 是 cupy，则索引必须也是 cupy
#             if isinstance(x_train, cp.ndarray):
#                 if isinstance(batch_idx, np.ndarray):
#                     batch_idx = cp.asarray(batch_idx)
#                 x_batch = x_train[batch_idx]
#                 t_batch = t_train[batch_idx]
#             else:
#                 x_batch = cp.asarray(
#                     x_train[batch_idx], dtype=cp.float32)
#                 t_batch = cp.asarray(
#                     t_train[batch_idx], dtype=cp.float32)

#             x_batch, t_batch = augment_batch(x_batch, t_batch)

#             loss = train_step(model, optimizer, x_batch, t_batch)
#             epoch_losses.append(loss)

#             del x_batch, t_batch
#             cp.get_default_memory_pool().free_all_blocks()

#             # 每几个批次记录一次损失值
#             batch_num = i // batch_size + 1
#             if batch_num % 10 == 0 or batch_num == n_batches:
#                 logger.info(
#                     f"[Epoch {epoch+1}/{epochs}] Batch {batch_num}/{n_batches} - Loss: {loss:.6f}")

#         # 添加当前epoch的平均损失
#         train_loss_list.append(sum(epoch_losses) / len(epoch_losses))
#         del epoch_losses  # 释放不再需要的列表

#         # 分批计算训练精度
#         train_acc = compute_accuracy_in_batches(
#             model, x_train, t_train, batch_size=batch_size)
#         train_acc_list.append(train_acc)

#         # 分批计算测试精度
#         if t_test is not None:
#             test_acc = compute_accuracy_in_batches(
#                 model, x_test, t_test, batch_size=batch_size)
#             test_acc_list.append(test_acc)
#             logger.info(
#                 f"End of Epoch {epoch+1} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

#             if test_acc > best_acc:
#                 best_acc = test_acc
#                 save_model(model, "best_model.npz")
#         else:
#             test_acc_list.append(None)
#             logger.info(f"End of Epoch {epoch+1} - Train Acc: {train_acc:.4f}")

#         # 每个epoch结束后释放内存
#         cp.get_default_memory_pool().free_all_blocks()

#     return model, train_loss_list, train_acc_list, test_acc_list

def train_model_optimized(model, x_train, t_train, x_test, t_test, epochs=10, batch_size=32, learning_rate=0.001):
    """优化内存使用的训练函数"""
    # 确保所有数据在GPU上，但采用较小批次
    optimizer = SGD(learning_rate)
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    best_acc = 0

    # 计算并记录总批次数，避免重复计算
    n_batches = (x_train.shape[0] + batch_size - 1) // batch_size

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        # 每个epoch清空临时变量以释放内存
        epoch_losses = []
        cp.get_default_memory_pool().free_all_blocks()

        # 分批处理数据以减少内存使用
        # 确保生成随机索引时使用与x_train相同的类型
        if isinstance(x_train, cp.ndarray):
            batch_indices = cp.random.permutation(x_train.shape[0])
        else:
            batch_indices = np.random.permutation(x_train.shape[0])

        for i in range(0, x_train.shape[0], batch_size):
            batch_idx = batch_indices[i:i+batch_size]

            # 处理索引和数组类型一致性问题
            if isinstance(x_train, cp.ndarray):
                # 如果 x_train 是 CuPy 数组，确保索引也是 CuPy 数组
                if isinstance(batch_idx, np.ndarray):
                    batch_idx = cp.asarray(batch_idx)
                x_batch = x_train[batch_idx]
                t_batch = t_train[batch_idx]
            else:
                # 如果 x_train 是 NumPy 数组，先提取数据，再转为 CuPy
                x_batch = cp.asarray(x_train[batch_idx], dtype=cp.float32)
                t_batch = cp.asarray(t_train[batch_idx], dtype=cp.float32)

            # 数据增强
            x_batch, t_batch = augment_batch(x_batch, t_batch)

            # 执行训练步骤
            loss = train_step(model, optimizer, x_batch, t_batch)
            epoch_losses.append(loss)

            # 释放内存
            del x_batch, t_batch
            cp.get_default_memory_pool().free_all_blocks()

            # 每几个批次记录一次损失值
            batch_num = i // batch_size + 1
            if batch_num % 10 == 0 or batch_num == n_batches:
                logger.info(
                    f"[Epoch {epoch+1}/{epochs}] Batch {batch_num}/{n_batches} - Loss: {loss:.6f}")

        # 添加当前epoch的平均损失
        train_loss_list.append(sum(epoch_losses) / len(epoch_losses))
        del epoch_losses  # 释放不再需要的列表

        # 分批计算训练精度
        train_acc = compute_accuracy_in_batches(
            model, x_train, t_train, batch_size=batch_size)
        train_acc_list.append(train_acc)

        # 分批计算测试精度
        if t_test is not None:
            test_acc = compute_accuracy_in_batches(
                model, x_test, t_test, batch_size=batch_size)
            test_acc_list.append(test_acc)
            logger.info(
                f"End of Epoch {epoch+1} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                save_model(model, "best_model.npz")
        else:
            test_acc_list.append(None)
            logger.info(f"End of Epoch {epoch+1} - Train Acc: {train_acc:.4f}")

        # 每个epoch结束后释放内存
        cp.get_default_memory_pool().free_all_blocks()

    return model, train_loss_list, train_acc_list, test_acc_list


def compute_accuracy_in_batches(model, x, t, batch_size=64):
    total_correct = 0
    total_samples = x.shape[0]

    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        x_batch_np = x[i:end_idx]
        t_batch_np = t[i:end_idx]

        # 只在当前 batch 转为 CuPy
        x_batch = cp.asarray(x_batch_np, dtype=cp.float32)
        t_batch = cp.asarray(t_batch_np, dtype=cp.float32)

        # 预测
        y = cp.argmax(model.predict(x_batch), axis=1)
        if t_batch.ndim != 1:
            t_batch = cp.argmax(t_batch, axis=1)

        total_correct += cp.sum(y == t_batch).item()

        # 显式释放 GPU 内存
        del x_batch, t_batch, y
        cp.get_default_memory_pool().free_all_blocks()

    return float(total_correct) / total_samples


# ===== 2. 优化im2col实现，减少内存使用 =====


def im2col_optimized(input_data, filter_h, filter_w, stride=1, pad=0):
    """优化的im2col，减少内存占用"""
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 采用更内存高效的方式处理padding
    if pad > 0:
        img = cp.pad(input_data, [(0, 0), (0, 0),
                     (pad, pad), (pad, pad)], 'constant')
    else:
        img = input_data

    # 直接创建输出数组而非中间数组
    col = cp.zeros((N * out_h * out_w, C * filter_h * filter_w),
                   dtype=input_data.dtype)

    # 内存友好的实现，减少中间数组的创建
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            for c in range(C):
                # 一次处理一个通道以减少内存需求
                values = img[:, c, y:y_max:stride,
                             x:x_max:stride].reshape(N, -1)
                col_idx = (c * filter_h + y) * filter_w + x
                col[:, col_idx] = values.ravel()

    return col

# ===== 3. 修改网络层以释放不需要的中间变量 =====


class ConvLayer:
    def __init__(self, in_channels, filter_num, filter_size, pad=0, stride=1):
        self.filter_num, self.pad, self.stride = filter_num, pad, stride
        scale = cp.sqrt(2.0 / (in_channels * filter_size * filter_size))
        self.params = {
            'W': cp.random.randn(filter_num, in_channels, filter_size, filter_size).astype(cp.float32) * scale,
            'b': cp.zeros(filter_num, dtype=cp.float32)
        }
        self.grads = {k: cp.zeros_like(v) for k, v in self.params.items()}
        self.cache = {}  # 使用缓存字典替代直接存储x和col

    def forward(self, x):
        if not isinstance(x, cp.ndarray):
            x = cp.asarray(x)

        FN, C, FH, FW = self.params['W'].shape
        N, _, H, W = x.shape
        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (W + 2*self.pad - FW) // self.stride + 1

        # 只保存必要的信息用于反向传播
        self.cache['x_shape'] = x.shape

        col = im2col_optimized(x, FH, FW, self.stride, self.pad)
        self.cache['col'] = col  # 保存在缓存中

        W_reshaped = self.params['W'].reshape(FN, -1).T
        out = cp.dot(col, W_reshaped) + self.params['b']

        return out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

    def backward(self, dout):
        if not isinstance(dout, cp.ndarray):
            dout = cp.asarray(dout)

        FN, C, FH, FW = self.params['W'].shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # 使用缓存中的col而不是保存完整的x
        col = self.cache.get('col')
        if col is None:
            raise ValueError("Forward must be called before backward")

        self.grads['W'] = cp.dot(col.T, dout).transpose(
            1, 0).reshape(FN, -1, FH, FW)
        self.grads['b'] = cp.sum(dout, axis=0)

        dcol = cp.dot(dout, self.params['W'].reshape(FN, -1))
        dx = col2im(dcol, self.cache['x_shape'], FH, FW, self.stride, self.pad)

        # 清理缓存以释放内存
        self.cache.clear()

        return dx


def plot_training_history(train_loss_list, train_acc_list, test_acc_list, save_path=None):
    # 避免不必要的转换 - 只在需要绘图时进行一次转换
    # 假设损失值和准确率已经是CPU上的数值或列表
    epochs = len(train_acc_list)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')

    # 确保数据在CPU上 - 一次性转换，而不是每个点都转换
    if hasattr(train_loss_list, 'get'):  # 检查是否是CuPy数组
        train_loss_np = cp.asnumpy(train_loss_list)
    else:
        train_loss_np = train_loss_list

    ax1.plot(cp.asnumpy(cp.array(train_loss_np)).reshape(-1, int(len(train_loss_np)/epochs)).mean(axis=1),
             color='tab:red', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')

    # 确保准确率数据在CPU上
    if hasattr(train_acc_list, 'get'):
        train_acc_np = cp.asnumpy(train_acc_list)
    else:
        train_acc_np = train_acc_list

    ax2.plot(train_acc_np, color='tab:blue', marker='o', label='Train Acc')

    if test_acc_list[0] is not None:
        if hasattr(test_acc_list, 'get'):
            test_acc_np = cp.asnumpy(test_acc_list)
        else:
            test_acc_np = test_acc_list
        ax2.plot(test_acc_np, color='tab:green', marker='x', label='Test Acc')

    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history saved to {save_path}")
    else:
        plt.show()

# ========== 辅助函数 ==========


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """优化版im2col，确保所有数据在GPU上"""
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 确保输入在GPU上
    if not isinstance(input_data, cp.ndarray):
        input_data = cp.asarray(input_data)

    img = cp.pad(input_data, [(0, 0), (0, 0),
                 (pad, pad), (pad, pad)], 'constant')
    col = cp.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=cp.float32)

    # 使用GPU核心计算，减少循环
    y_indices = cp.arange(filter_h)
    x_indices = cp.arange(filter_w)

    for y in y_indices:
        y_max = y + stride * out_h
        for x in x_indices:
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """优化版col2im，确保所有数据在GPU上"""
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 确保输入在GPU上
    if not isinstance(col, cp.ndarray):
        col = cp.asarray(col)

    col = col.reshape(N, out_h, out_w, C, filter_h,
                      filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = cp.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cp.float32)

    # 使用GPU核心计算，减少循环
    y_indices = cp.arange(filter_h)
    x_indices = cp.arange(filter_w)

    for y in y_indices:
        y_max = y + stride * out_h
        for x in x_indices:
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad] if pad > 0 else img


def softmax(x):
    """确保数据在GPU上的softmax计算"""
    if not isinstance(x, cp.ndarray):
        x = cp.asarray(x)

    x = x - x.max(axis=1, keepdims=True) if x.ndim == 2 else x - x.max()
    exp_x = cp.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True) if x.ndim == 2 else exp_x / exp_x.sum()


def cross_entropy_error(y, t):
    """确保数据在GPU上的交叉熵计算"""
    if not isinstance(y, cp.ndarray):
        y = cp.asarray(y)
    if not isinstance(t, cp.ndarray):
        t = cp.asarray(t)

    if y.ndim == 1:
        t, y = t.reshape(1, -1), y.reshape(1, -1)
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -cp.sum(cp.log(y[cp.arange(batch_size), t] + 1e-7)) / batch_size


# ========== 基础层 ==========
# class ConvLayer:
#     def __init__(self, in_channels, filter_num, filter_size, pad=0, stride=1):
#         self.filter_num, self.pad, self.stride = filter_num, pad, stride
#         # 使用Xavier初始化，而不是固定的0.01缩放
#         scale = cp.sqrt(2.0 / (in_channels * filter_size * filter_size))
#         self.params = {
#             'W': cp.random.randn(filter_num, in_channels, filter_size, filter_size).astype(cp.float32) * scale,
#             'b': cp.zeros(filter_num, dtype=cp.float32)
#         }
#         self.grads = {k: cp.zeros_like(v) for k, v in self.params.items()}
#         self.x = None
#         self.col = None

#     def forward(self, x):
#         # 确保输入在GPU上
#         if not isinstance(x, cp.ndarray):
#             x = cp.asarray(x)

#         FN, C, FH, FW = self.params['W'].shape
#         N, _, H, W = x.shape
#         out_h = (H + 2*self.pad - FH) // self.stride + 1
#         out_w = (W + 2*self.pad - FW) // self.stride + 1

#         col = im2col(x, FH, FW, self.stride, self.pad)
#         out = cp.dot(col, self.params['W'].reshape(
#             FN, -1).T) + self.params['b']
#         self.x, self.col = x, col
#         return out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

#     def backward(self, dout):
#         # 确保输入在GPU上
#         if not isinstance(dout, cp.ndarray):
#             dout = cp.asarray(dout)

#         FN, _, FH, FW = self.params['W'].shape
#         dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

#         self.grads['W'] = cp.dot(self.col.T, dout).transpose(
#             1, 0).reshape(FN, -1, FH, FW)
#         self.grads['b'] = cp.sum(dout, axis=0)

#         dcol = cp.dot(dout, self.params['W'].reshape(FN, -1))
#         return col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)


class PoolingLayer:
    def __init__(self, pool_size, stride=2):
        self.pool_size, self.stride = pool_size, stride
        self.x = None
        self.arg_max = None

    def forward(self, x):
        # 确保输入在GPU上
        if not isinstance(x, cp.ndarray):
            x = cp.asarray(x)

        self.x = x
        N, C, H, W = x.shape
        out_h, out_w = (H - self.pool_size) // self.stride + \
            1, (W - self.pool_size) // self.stride + 1

        col = im2col(x, self.pool_size, self.pool_size, self.stride, 0)
        col = col.reshape(N * out_h * out_w * C, -1)

        self.arg_max = cp.argmax(col, axis=1)
        out = cp.max(col, axis=1)

        return out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

    def backward(self, dout):
        # 确保输入在GPU上
        if not isinstance(dout, cp.ndarray):
            dout = cp.asarray(dout)

        N, C, H, W = self.x.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        dout = dout.transpose(0, 2, 3, 1).flatten()
        dmax = cp.zeros((dout.size, self.pool_size *
                        self.pool_size), dtype=cp.float32)
        dmax[cp.arange(dout.size), self.arg_max] = dout

        dcol = dmax.reshape(N * out_h * out_w * C, -1)
        dx = col2im(dcol, self.x.shape, self.pool_size,
                    self.pool_size, self.stride, 0)

        return dx


class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        # 使用Xavier初始化
        scale = cp.sqrt(2.0 / input_size)
        self.params = {
            'W': cp.random.randn(input_size, output_size).astype(cp.float32) * scale,
            'b': cp.zeros(output_size, dtype=cp.float32)
        }
        self.grads = {k: cp.zeros_like(v) for k, v in self.params.items()}
        self.x = None

    def forward(self, x):
        # 确保输入在GPU上
        if not isinstance(x, cp.ndarray):
            x = cp.asarray(x)

        self.x = x
        return cp.dot(x, self.params['W']) + self.params['b']

    def backward(self, dout):
        # 确保输入在GPU上
        if not isinstance(dout, cp.ndarray):
            dout = cp.asarray(dout)

        self.grads['W'] = cp.dot(self.x.T, dout)
        self.grads['b'] = cp.sum(dout, axis=0)

        return cp.dot(dout, self.params['W'].T)


class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # 确保输入在GPU上
        if not isinstance(x, cp.ndarray):
            x = cp.asarray(x)

        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # 确保输入在GPU上
        if not isinstance(dout, cp.ndarray):
            dout = cp.asarray(dout)

        dout = dout.copy()  # 避免修改原始数据
        dout[self.mask] = 0
        return dout


class FlattenLayer:
    def __init__(self):
        self.orig_shape = None

    def forward(self, x):
        # 确保输入在GPU上
        if not isinstance(x, cp.ndarray):
            x = cp.asarray(x)

        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        # 确保输入在GPU上
        if not isinstance(dout, cp.ndarray):
            dout = cp.asarray(dout)

        return dout.reshape(self.orig_shape)


class SoftmaxWithLoss:
    def __init__(self):
        self.t = None
        self.y = None
        self.loss = None

    def forward(self, x, t):
        # 确保输入在GPU上
        if not isinstance(x, cp.ndarray):
            x = cp.asarray(x)
        if not isinstance(t, cp.ndarray):
            t = cp.asarray(t)

        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size


class LeakyReLULayer:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.input_data = None

    def forward(self, input_data):
        self.input_data = input_data
        return cp.where(input_data > 0, input_data, self.alpha * input_data)

    def backward(self, output_gradient):
        """
        Backward pass for LeakyReLU preserving dimensions
        """
        # Handle 1D gradient array
        if len(output_gradient.shape) == 1:
            # Reshape to match input dimensions
            batch_size = self.input_data.shape[0]
            output_gradient = cp.tile(
                output_gradient.reshape(1, -1), (batch_size, 1))

        # Calculate gradient multiplier
        d_input = cp.where(self.input_data > 0, 1, self.alpha)

        # Ensure dimensions match
        if d_input.shape != output_gradient.shape:
            if len(d_input.shape) > len(output_gradient.shape):
                # Reshape output_gradient to match d_input
                output_gradient = cp.broadcast_to(
                    output_gradient, d_input.shape)
            else:
                # Reshape d_input to match output_gradient if needed
                d_input = cp.broadcast_to(d_input, output_gradient.shape)

        return output_gradient * d_input


# ========== 简单CNN网络 ==========
class SimpleCNN:
    def __init__(self, input_dim, output_size):
        self.layers = [
            ConvLayer(input_dim[0], 32, 3,
                      pad=1), LeakyReLULayer(), PoolingLayer(2),
            ConvLayer(32, 64, 3, pad=1), LeakyReLULayer(), PoolingLayer(2),
            ConvLayer(64, 128, 3, pad=1), LeakyReLULayer(), PoolingLayer(2),
            FlattenLayer()
        ]
        # 使用GPU上的dummy输入来计算全连接层的尺寸
        dummy_x = cp.zeros((1, *input_dim), dtype=cp.float32)
        for layer in self.layers:
            dummy_x = layer.forward(dummy_x)

        fc_input_size = dummy_x.size // dummy_x.shape[0]
        self.layers += [
            FullyConnectedLayer(fc_input_size, 256), LeakyReLULayer(),
            FullyConnectedLayer(256, 128), LeakyReLULayer(),
            FullyConnectedLayer(128, output_size)
        ]
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        # 确保输入在GPU上
        if not isinstance(x, cp.ndarray):
            x = cp.asarray(x, dtype=cp.float32)

        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        # 确保输入在GPU上
        if not isinstance(x, cp.ndarray):
            x = cp.asarray(x, dtype=cp.float32)
        if not isinstance(t, cp.ndarray):
            t = cp.asarray(t, dtype=cp.float32)

        return self.last_layer.forward(self.predict(x), t)

    def accuracy(self, x, t, batch_size=64):
        # 确保输入在GPU上
        if not isinstance(x, cp.ndarray):
            x = cp.asarray(x, dtype=cp.float32)
        if not isinstance(t, cp.ndarray):
            t = cp.asarray(t, dtype=cp.float32)

        acc = 0
        # 使用GPU上的计数器避免反复传输数据
        for i in range(0, x.shape[0], batch_size):
            x_batch, t_batch = x[i:i+batch_size], t[i:i+batch_size]
            y = cp.argmax(self.predict(x_batch), axis=1)
            if t_batch.ndim != 1:
                t_batch = cp.argmax(t_batch, axis=1)
            acc += cp.sum(y == t_batch)

        return float(acc) / float(x.shape[0])  # 只在最后转换一次

    def gradient(self, x, t):
        # 确保输入在GPU上
        if not isinstance(x, cp.ndarray):
            x = cp.asarray(x, dtype=cp.float32)
        if not isinstance(t, cp.ndarray):
            t = cp.asarray(t, dtype=cp.float32)

        self.loss(x, t)
        dout = self.last_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return {k: v for layer in self.layers if hasattr(layer, 'grads') for k, v in layer.grads.items()}


# ========== 训练相关 ==========
class SGD:
    def __init__(self, lr=0.001):
        self.lr = lr

    def update(self, layers):
        for layer in layers:
            if hasattr(layer, 'params'):
                for k in layer.params:
                    layer.params[k] -= self.lr * layer.grads[k]


def save_model(model, path):
    # 保存模型参数到CPU内存，然后保存到磁盘
    params = {}
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, 'params'):
            for k, v in layer.params.items():
                # 转换为NumPy数组以保存
                params[f'layer{idx}_{k}'] = cp.asnumpy(v)

    # 使用NumPy保存
    np.savez(path, **params)


def load_model(model, path):
    # 加载模型参数并转移到GPU
    data = np.load(path)

    for idx, layer in enumerate(model.layers):
        if hasattr(layer, 'params'):
            for k in layer.params:
                # 确保参数在GPU上
                layer.params[k] = cp.asarray(data[f'layer{idx}_{k}'])


def train_step(model, optimizer, x_batch, t_batch):
    # 确保批次数据在GPU上
    if not isinstance(x_batch, cp.ndarray):
        x_batch = cp.asarray(x_batch, dtype=cp.float32)
    if not isinstance(t_batch, cp.ndarray):
        t_batch = cp.asarray(t_batch, dtype=cp.float32)

    model.gradient(x_batch, t_batch)
    optimizer.update(model.layers)
    return float(model.loss(x_batch, t_batch))  # 返回Python浮点数


def batch_generator(x, t, batch_size, shuffle=True):
    """批次生成器函数，减少GPU-CPU数据传输"""
    n_samples = x.shape[0]

    if shuffle:
        # 只计算一次索引顺序并保存在GPU上
        indices = cp.random.permutation(n_samples)
        x_shuffled = x[indices]
        t_shuffled = t[indices]
    else:
        x_shuffled, t_shuffled = x, t

    # 一次生成一批数据
    for i in range(0, n_samples, batch_size):
        x_batch = x_shuffled[i:i + batch_size]
        t_batch = t_shuffled[i:i + batch_size]
        yield x_batch, t_batch


def train_model(model, x_train, t_train, x_test, t_test, epochs=10, batch_size=32, learning_rate=0.001):
    """优化的训练函数，减少GPU-CPU数据传输"""
    # 确保所有数据在GPU上
    if not isinstance(x_train, cp.ndarray):
        x_train = cp.asarray(x_train, dtype=cp.float32)
    if not isinstance(t_train, cp.ndarray):
        t_train = cp.asarray(t_train, dtype=cp.float32)

    if x_test is not None and not isinstance(x_test, cp.ndarray):
        x_test = cp.asarray(x_test, dtype=cp.float32)
    if t_test is not None and not isinstance(t_test, cp.ndarray):
        t_test = cp.asarray(t_test, dtype=cp.float32)

    optimizer = SGD(learning_rate)
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    best_acc = 0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        # 创建批次生成器
        batch_gen = batch_generator(x_train, t_train, batch_size, shuffle=True)
        n_batches = x_train.shape[0] // batch_size

        # 跟踪当前epoch的损失
        epoch_losses = []

        for i, (x_batch, t_batch) in enumerate(batch_gen):
            # 数据增强 - 确保增强后数据仍在GPU上
            x_batch, t_batch = augment_batch(x_batch, t_batch)

            # 执行一步训练
            loss = train_step(model, optimizer, x_batch, t_batch)
            epoch_losses.append(loss)

            # 每个批次后记录损失值
            if (i+1) % 10 == 0 or (i+1) == n_batches:  # 每10批次或最后一批
                logger.info(
                    f"[Epoch {epoch + 1}/{epochs}] Batch {i + 1}/{n_batches} - Loss: {loss:.6f}")

        # 扩展损失历史记录
        train_loss_list.extend(epoch_losses)

        # 计算训练和测试准确率（保持在GPU上）
        train_acc = model.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)

        if t_test is not None:
            test_acc = model.accuracy(x_test, t_test)
            test_acc_list.append(test_acc)
            logger.info(
                f"End of Epoch {epoch + 1} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                save_model(model, "best_model.npz")
        else:
            test_acc_list.append(None)
            logger.info(
                f"End of Epoch {epoch + 1} - Train Acc: {train_acc:.4f}")

    # 将GPU上的结果转移到CPU，以便调用者处理
    return model, train_loss_list, train_acc_list, test_acc_list


# # ========== main 测试 ==========
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     # 创建GPU上的随机测试数据
#     input_dim, output_size = (3, 128, 128), 2
#     x_train, t_train, x_test, t_test, class_names = load_and_preprocess_data(
#         r'D:\QQ下载\DEEPlearning\D_\DEEPlearning\data')
#     # 创建模型并训练
#     model = SimpleCNN(input_dim, output_size)
#     model, train_loss_list, train_acc_list, test_acc_list = train_model(
#         model, x_train, t_train, x_test, t_test, epochs=10, batch_size=16, learning_rate=0.01
#     )

#     # 显示最终结果
#     print(f"Final Train Accuracy: {train_acc_list[-1]:.4f}")

#     # 绘制训练历史
#     plot_training_history(train_loss_list, train_acc_list,
#                           test_acc_list, save_path="training_history.png")


# ========== 主程序中添加内存管理 ==========
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 设置较小的批次大小
    batch_size = 8

    # 加载数据（保留在 CPU 上）
    x_train, t_train, x_test, t_test, class_names = load_and_preprocess_data('data')

    # 获取输入维度和输出类别数
    input_dim = x_train.shape[1:]
    output_size = t_train.shape[1] if t_train.ndim > 1 else len(
        np.unique(t_train))

    # 创建模型
    model = SimpleCNN(input_dim, output_size)

    # 只将数据在每个 batch 内转为 cupy 并训练
    model, train_loss_list, train_acc_list, test_acc_list = train_model_optimized(
        model, x_train, t_train, x_test, t_test,
        epochs=10, batch_size=batch_size, learning_rate=0.01
    )

    print(f"Final Train Accuracy: {train_loss_list[-1]:.4f}")

    # 保存训练历史
    np.savez('training_history.npz',
             train_loss_list=train_loss_list,
             train_acc_list=train_acc_list,
             test_acc_list=test_acc_list)
    print('训练历史已保存为 training_history.npz')

    # 绘图前转换为 numpy
    from visualizer import plot_loss_and_accuracy
    plot_loss_and_accuracy(train_loss_list, train_acc_list, test_acc_list)
