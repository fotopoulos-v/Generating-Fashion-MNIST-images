# app.py
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from models import vae_decoder

latent_dim = 10 

app = Flask(__name__)



def encode_image(image_tensor):
    image_array = (image_tensor.numpy().squeeze() * 127.5 + 127.5).astype(np.uint8)
    image = Image.fromarray(image_array, mode="L").resize((280, 280))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.route("/", methods=["GET"])
def index():
    return render_template("index-project3.html")

@app.route("/generate/<model_type>", methods=["GET"])
def generate(model_type):
    noise = tf.random.normal([1, latent_dim])

    if model_type == "vae":
        image = vae_decoder(noise, training=False)
        img_b64 = encode_image(image)
    # elif model_type == "dcgan":
    #     image = dcgan_generator(noise, training=False)
    #     img_b64 = encode_image(image)
    # elif  model_type == "gan":
    #     image = gan_generator(noise, training=False)
    #     img_b64 = encode_image(image)
    # elif model_type == "diffusion":
    #     images, _, _ = sample_diffusion(diffusion_model, batch_size=1)
    #     img_b64 = encode_image(tf.convert_to_tensor(images[0]))
    else:
        return jsonify({"error": "Invalid model type"}), 400

    return jsonify({"image": img_b64})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
