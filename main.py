from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import io
import numpy as np
import torchvision.transforms as T
import torch
import os

main = Flask(__name__)

# Load the model
learn_inf = torch.jit.load("checkpoints/transfer_exported.pt")

@main.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file:
            # Load the image
            img = Image.open(uploaded_file)
            img.load()

            # Save the uploaded image to a temporary location
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            img_path = os.path.join(upload_dir, "uploaded_image.jpg")
            img.save(img_path)

            # Transform to tensor
            timg = T.ToTensor()(img).unsqueeze_(0)

            # Calling the model
            softmax = learn_inf(timg).data.cpu().numpy().squeeze()

            # Get the indexes of the classes ordered by softmax
            idxs = np.argsort(softmax)[::-1]

            # Get the top 5 predictions
            top_classes = [learn_inf.class_names[idx] for idx in idxs[:5]]
            top_probs = [float(softmax[idx]) for idx in idxs[:5]]

            # Calculate the maximum probability
            max_prob = max(top_probs)

            # Pass image_path to the template
            return render_template("index.html", classes=top_classes, probs=top_probs, image_path=img_path, max_prob=max_prob)

    return render_template("index.html", classes=None, probs=None, image_path=None)

@main.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory("uploads", filename)

if __name__ == "__main__":
    main.run(debug=True)