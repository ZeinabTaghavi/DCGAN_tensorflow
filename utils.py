import tensorflow as tf


def load_dataset_mnist(img_shape,buffer_size,batches):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], img_shape[0], img_shape[1], img_shape[2]).astype('float32')

    train_images = (train_images - 127.5) / 127.5

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batches)
    
    return train_dataset