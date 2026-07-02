import tensorflow as tf
print("NumPy 版本:", tf.__version__)
print("GPU 是否可用:", tf.test.is_gpu_available())
print("可用的 GPU 裝置:", tf.config.list_physical_devices('GPU'))