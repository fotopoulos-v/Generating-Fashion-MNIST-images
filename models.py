# models.py
import tensorflow as tf

codings_size = 10
decoder_inputs = tf.keras.Input(shape=(codings_size,))
x = tf.keras.layers.Dense(100, activation="relu", name="dense")(decoder_inputs)
x = tf.keras.layers.Dense(150, activation="relu", name="dense_1")(x)
x = tf.keras.layers.Dense(28 * 28, name="dense_2")(x)
outputs = tf.keras.layers.Reshape([28, 28])(x)

vae_decoder = tf.keras.Model(decoder_inputs, outputs)
vae_decoder(tf.zeros((1, codings_size)))  # builds the model by running a dummy input
vae_decoder.load_weights("static/assets/keras/vae_decoder.weights.h5")



# import tensorflow as tf
# import numpy as np
# import time

# # Define the latent dimension used in the decoder
# codings_size = 10  # Must match what was used during training

# # Rebuild the decoder exactly as in your training script
# decoder_inputs = tf.keras.layers.Input(shape=[codings_size])
# x = tf.keras.layers.Dense(100, activation="relu", name="dense")(decoder_inputs)
# x = tf.keras.layers.Dense(150, activation="relu", name="dense_1")(x)
# x = tf.keras.layers.Dense(28 * 28, name="dense_2")(x)
# outputs = tf.keras.layers.Reshape([28, 28], name="reshape")(x)

# vae_decoder = tf.keras.Model(inputs=decoder_inputs, outputs=outputs)
# dummy = tf.random.normal([1, codings_size])
# _ = vae_decoder(dummy)  # build the model by calling once
# # Load the weights (not the full model!)
# vae_decoder.load_weights('/home/vf2/Desktop/Bill_site_git/portfolio/static/assets/keras/vae_decoder.weights.h5')

# # Load all trained models (adjust paths to yours)

# # gan_generator = tf.keras.models.load_model('/home/vf2/Desktop/Bill_site_git/portfolio/static/assets/keras/gan_generator.h5')
# # dcgan_generator = tf.keras.models.load_model('/home/vf2/Desktop/Bill_site_git/portfolio/static/assets/keras/dcgan_generator.h5')
# # diffusion_model = tf.keras.models.load_model('/home/vf2/Desktop/Bill_site_git/portfolio/static/assets/keras/diffusion_model_generator.h5')

# # # Variance schedule for diffusion
# # def variance_schedule(T, s=0.008, max_beta=0.999):
# #     t = np.arange(T + 1)
# #     f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
# #     alpha = np.clip(f[1:] / f[:-1], 1 - max_beta, 1)
# #     alpha = np.append(1, alpha).astype(np.float32)
# #     beta = 1 - alpha
# #     alpha_cumprod = np.cumprod(alpha)
# #     return alpha, alpha_cumprod, beta

# # T = 4000
# # alpha, alpha_cumprod, beta = variance_schedule(T)

# # # Sampling function for diffusion
# # def sample_diffusion(model, batch_size=32, T=T):
# #     X = tf.random.normal([batch_size, 28, 28, 1])
# #     start_time = time.time()

# #     for t in range(T - 1, 0, -1):
# #         noise = tf.random.normal(tf.shape(X)) if t > 1 else tf.zeros(tf.shape(X))
# #         X_noise = model({"X_noisy": X, "time": tf.constant([t] * batch_size)})
# #         X = (
# #             1 / np.sqrt(alpha[t]) *
# #             (X - beta[t] / np.sqrt(1 - alpha_cumprod[t]) * X_noise) +
# #             np.sqrt(1 - alpha[t]) * noise
# #         )

# #     exec_time = time.time() - start_time
# #     return X.numpy(), exec_time, T
