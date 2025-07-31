# app.py
from flask import Flask, request, jsonify
from models import vae_decoder
import numpy as np

app = Flask(__name__)

from flask import Flask, request, jsonify, render_template
# ⬆ make sure render_template is imported

@app.route("/", methods=["GET"])
def index():
    return render_template("index-project3.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Receive latent vector
        latent = request.json.get("latent")
        if not latent or len(latent) != 10:
            return jsonify({"error": "Expected 'latent' array of length 10"}), 400

        z = np.array(latent).reshape(1, -1)
        image = vae_decoder.predict(z)[0].tolist()
        return jsonify({"image": image})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
