import numpy as np
from visualizer import plot_loss_and_accuracy

data = np.load('training_history.npz')
train_loss_list = data['train_loss_list']
train_acc_list = data['train_acc_list']
test_acc_list = data['test_acc_list']

plot_loss_and_accuracy(train_loss_list, train_acc_list, test_acc_list)
print('已根据历史数据重新绘制 training_history.png') 