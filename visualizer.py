import matplotlib.pyplot as plt
import numpy as np

def to_numpy_list(arr):
    # 支持 list/array/标量的递归转换
    if hasattr(arr, 'get'):
        return arr.get()
    elif isinstance(arr, (list, tuple)):
        return [to_numpy_list(a) for a in arr]
    else:
        return arr

def plot_loss_and_accuracy(train_loss_list, train_acc_list, test_acc_list):
    # 转换为 numpy 可用的数据
    train_loss_list = to_numpy_list(train_loss_list)
    train_acc_list = to_numpy_list(train_acc_list)
    test_acc_list = to_numpy_list(test_acc_list)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Training Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.savefig('training_history.png')
    print('训练过程曲线已保存为 training_history.png')