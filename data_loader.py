import os
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger("DataLoader")


def load_and_preprocess_data(data_dir='data', img_size=(128, 128)):
    # 自动适配为相对于当前脚本的绝对路径
    if not os.path.isabs(data_dir):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(base_dir, '..', data_dir))
    logger.info(f"Loading and preprocessing data from {data_dir} ...")
    class_names = ['cat', 'dog']
    x_train, t_train = [], []
    x_test, t_test = [], []

    def process_image(path):
        img = Image.open(path).resize(img_size)
        img_array = np.array(img) / 255.0
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=2)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        return img_array.transpose(2, 0, 1).astype(np.float32)

    # 训练数据
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, 'train', class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_array = process_image(
                        os.path.join(class_path, img_name))
                    x_train.append(img_array)
                    label = np.zeros(len(class_names), dtype=np.float32)
                    label[class_idx] = 1
                    t_train.append(label)
                except Exception as e:
                    logger.warning(f"Failed loading {img_name}: {e}")

    # 测试数据
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, 'test', class_name)
        if not os.path.exists(class_path):
            continue
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_array = process_image(
                        os.path.join(class_path, img_name))
                    x_test.append(img_array)
                    label = np.zeros(len(class_names), dtype=np.float32)
                    label[class_idx] = 1
                    t_test.append(label)
                except Exception as e:
                    logger.warning(f"Failed loading {img_name}: {e}")

    # 转为 NumPy 数组（留在 CPU 上）
    x_train, t_train = np.array(x_train), np.array(t_train)
    x_test, t_test = np.array(x_test), np.array(t_test)

    # 打乱数据
    idx = np.random.permutation(len(x_train))
    x_train, t_train = x_train[idx], t_train[idx]
    idx = np.random.permutation(len(x_test))
    x_test, t_test = x_test[idx], t_test[idx]

    logger.info(
        f"Loaded {len(x_train)} training and {len(x_test)} test samples.")
    return x_train, t_train, x_test, t_test, class_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    x_train, t_train, x_test, t_test, class_names = load_and_preprocess_data('data')
    print(
        f"Train: {x_train.shape}, {t_train.shape} | Test: {x_test.shape}, {t_test.shape}")
