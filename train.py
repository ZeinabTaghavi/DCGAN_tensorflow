import tensorflow as tf
from models import Generator , G_loss , Discriminator , D_loss
import matplotlib.pyplot as plt


import os

def train(train_dataset,g_lr = 1e-4,
         d_lr = 1e-4,
         epoches = 1,
         g_batch_size = 16,
         noise_dim = 100):
    
    G = Generator()
    D = Discriminator()
    G_optimizer = tf.keras.optimizers.Adam(g_lr)
    D_optimizer = tf.keras.optimizers.Adam(d_lr)
    
    tf.keras.utils.plot_model(G, to_file='./schema/Generator.png')
    tf.keras.utils.plot_model(D, to_file='./schema/Discriminator.png')
    
    
    num_examples_to_generate = 16
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    if os.path.exists('Gen_weights.h5'):
        G.load_weights('Gen_weights.h5')
    if os.path.exists('Dis_weights.h5'):
        D.load_weights('Dis_weights.h5')
       
    for epoch in range(epoches):
        i = 0
        for images in train_dataset:
            print(str(epoch)+' - '+str(i))
            i += 1
            noise = tf.random.normal([g_batch_size , noise_dim])

            with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
                generated_images = G(noise, training=True)

                real_output = D(images, training=True)
                fake_output = D(generated_images, training=True)

                gen_loss = G_loss(fake_output,cross_entropy)
                disc_loss = D_loss(real_output, fake_output,cross_entropy)

            gradients_of_generator = G_tape.gradient(gen_loss, G.trainable_variables)
            gradients_of_discriminator = D_tape.gradient(disc_loss, D.trainable_variables)

            G_optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables))
            D_optimizer.apply_gradients(zip(gradients_of_discriminator, D.trainable_variables))

        if epoch%10==0:
            G.save_weights('Gen_weights.h5')
            D.save_weights('Dis_weights.h5')

            for i in range(generated_images.shape[0]):
                plt.subplot(4, 4, i+1)
                plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
                plt.axis('off')

            plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
            plt.show()

    G.save_weights('Gen_weights.h5')
    D.save_weights('Dis_weights.h5')
