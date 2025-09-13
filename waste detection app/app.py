import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for plotting

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, send_file, Response
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from fpdf import FPDF
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -----------------------------
# Flask Config
# -----------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# -----------------------------
# Load Keras classification model
best_model = load_model(r"C:\Users\ASHAN\Desktop\waste detection model\waste detection app\efficientnet_b0_best.keras")
IMG_SIZE = 128

CLASS_LABELS = ['biological', 'brown-glass', 'cardboard', 'green-glass',
                'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

RECYCLABLE = ["brown-glass", "green-glass", "white-glass", "metal", "plastic", "paper", "cardboard"]
NON_RECYCLABLE = ["trash", "biological", "shoes"]

stats = {}

# -----------------------------
# Load YOLOv8 model
yolo_model = YOLO(r"C:\Users\ASHAN\Desktop\waste detection model\waste detection app\best.pt")

# -----------------------------
# Preprocess image for classification
def preprocess_image(file_path):
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    img_input = preprocess_input(img_rgb.astype("float32"))
    img_input = np.expand_dims(img_input, axis=0)
    return img_rgb, img_input

# -----------------------------
# Log predictions
def log_prediction(class_label):
    stats[class_label] = stats.get(class_label, 0) + 1
    total_items = sum(stats.values())
    with open("waste_log.csv", "w") as f:
        f.write("Waste Classification Report\n")
        f.write(f"Total Items Processed: {total_items}\n")
        for category, count in stats.items():
            f.write(f"{category}: {count}\n")

# -----------------------------
# Generate PDF report
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
# Image classification route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(request.url)

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        img_rgb, img_input = preprocess_image(file_path)
        preds = best_model.predict(img_input)
        class_idx = np.argmax(preds, axis=1)[0]
        class_label = CLASS_LABELS[class_idx]
        confidence = preds[0][class_idx]

        log_prediction(class_label)

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

# -----------------------------
# Show statistics
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
        plt.savefig("static/stats_chart.png")
        plt.close()
    return render_template("report.html", stats=stats)

# -----------------------------
# Download reports
@app.route("/download_pdf")
def download_pdf():
    pdf_path = generate_pdf_report()
    return send_file(pdf_path, as_attachment=True)

@app.route("/download_csv")
def download_csv():
    return send_file("waste_log.csv", as_attachment=True)

# -----------------------------
# Real-time camera detection
@app.route("/camera")
def camera():
    return render_template("camera.html")

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = yolo_model(frame)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = yolo_model.names[class_ids[i]]
            confidence = confidences[i]

            # 🎨 Color based on confidence
            if confidence >= 0.80:
                color = (0, 255, 0)   # Green
            elif confidence >= 0.50:
                color = (0, 255, 255) # Yellow
            else:
                color = (0, 0, 255)   # Red

            # Draw box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence*100:.1f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ✅ Log only high-confidence detections
            if confidence >= 0.80:
                log_prediction(label)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------
# Run Flask
if __name__ == "__main__":
    app.run(debug=True)
