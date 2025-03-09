import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
print("Is TensorFlow built with CUDA?", tf.test.is_built_with_cuda())
print("Is TensorFlow built with GPU support?", tf.test.is_built_with_gpu_support())

print(tf.sysconfig.get_build_info()["cuda_version"])
print(tf.sysconfig.get_build_info()["cudnn_version"])

