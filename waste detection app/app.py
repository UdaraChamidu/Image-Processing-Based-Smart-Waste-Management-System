import matplotlib
matplotlib.use('Agg')  # <- Use non-GUI backend

import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, send_file
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from fpdf import FPDF
import matplotlib.pyplot as plt

# -----------------------------
# Flask Config
# -----------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# -----------------------------
# Load Keras model
best_model = load_model(r"C:\Users\ASHAN\Desktop\waste detection model\waste detection app\efficientnet_b0_best.keras")

IMG_SIZE = 128

# Class labels (no label encoder needed)
CLASS_LABELS = ['biological', 'brown-glass', 'cardboard', 'green-glass', 
                'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# Waste categories
RECYCLABLE = ["brown-glass", "green-glass", "white-glass", "metal", "plastic", "paper", "cardboard"]
NON_RECYCLABLE = ["trash", "biological", "shoes"]

# Initialize statistics
stats = {}

# -----------------------------
# Preprocess image
def preprocess_image(file_path):
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    img_input = preprocess_input(img_rgb.astype("float32"))
    img_input = np.expand_dims(img_input, axis=0)
    return img_rgb, img_input

# -----------------------------
# Log predictions
def log_prediction(class_label):
    log_df = pd.DataFrame([[datetime.now(), class_label]], columns=["Timestamp", "Class"])
    try:
        old_df = pd.read_csv("waste_log.csv")
        new_df = pd.concat([old_df, log_df], ignore_index=True)
    except FileNotFoundError:
        new_df = log_df
    new_df.to_csv("waste_log.csv", index=False)

# -----------------------------
# PDF Report
def generate_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Waste Classification Report", ln=True, align="C")
    pdf.ln(10)

    total_items = sum(stats.values())
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Total Items Processed: {total_items}", ln=True)

    for category, count in stats.items():
        pdf.cell(0, 10, txt=f"{category}: {count}", ln=True)

    pdf_file = "waste_report.pdf"
    pdf.output(pdf_file)
    return pdf_file

# -----------------------------
# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            return redirect(request.url)

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Preprocess & predict
        img_rgb, img_input = preprocess_image(file_path)
        preds = best_model.predict(img_input)
        class_idx = np.argmax(preds, axis=1)[0]
        class_label = CLASS_LABELS[class_idx]
        confidence = preds[0][class_idx]

        # Update stats & log
        stats[class_label] = stats.get(class_label, 0) + 1
        log_prediction(class_label)

        # Determine bin type
        if class_label in RECYCLABLE:
            bin_type = "Recyclable ♻️"
        elif class_label in NON_RECYCLABLE:
            bin_type = "Non-Recyclable 🗑️"
        else:
            bin_type = "Unknown ⚠️"

        return render_template(
            "result.html",
            image=file.filename,
            label=class_label,
            confidence=f"{confidence*100:.2f}%",
            bin_type=bin_type
        )

    return render_template("index.html")


@app.route("/stats")
def show_stats():
    if stats:
        categories = list(stats.keys())
        counts = list(stats.values())
        plt.figure(figsize=(6, 4))
        plt.bar(categories, counts, color="green")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.title("Waste Classification Statistics")
        plt.tight_layout()
        plt.savefig("static/stats_chart.png")  # Safe with Agg backend
        plt.close()
    return render_template("report.html", stats=stats)


@app.route("/download_pdf")
def download_pdf():
    pdf_path = generate_pdf_report()
    return send_file(pdf_path, as_attachment=True)


@app.route("/download_csv")
def download_csv():
    return send_file("waste_log.csv", as_attachment=True)

# -----------------------------
# Run Flask
if __name__ == "__main__":
    app.run(debug=True)
