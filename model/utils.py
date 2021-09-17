import tensorflow as tf

def normalize(x, mean, std):
    mean = tf.constant(mean)
    std = tf.constant(std)
    return (x - mean) / std
    