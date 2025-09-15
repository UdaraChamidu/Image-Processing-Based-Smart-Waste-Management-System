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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# -----------------------------
# Load Keras classification model
try:
    best_model = load_model("efficientnet_b0_best.keras")
    print("✅ EfficientNet model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading EfficientNet model: {e}")
    best_model = None

# -----------------------------
# Load YOLO model
try:
    yolo_model = YOLO("best.pt")
    print("✅ YOLO model loaded successfully!")
    print(f"YOLO model classes: {yolo_model.names}")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    yolo_model = None

IMG_SIZE = 128

# EfficientNet classes
CLASS_LABELS = ['biological', 'brown-glass', 'cardboard', 'green-glass',
                'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

RECYCLABLE = ["brown-glass", "green-glass", "white-glass", "metal", "plastic", "paper", "cardboard"]
NON_RECYCLABLE = ["trash", "biological", "shoes"]

stats = {}

# YOLO detection confidence threshold
YOLO_CONFIDENCE_THRESHOLD = 0.5

# -----------------------------
# YOLO object detection function
def detect_objects_yolo(file_path):
    """Detect objects in image using YOLO model"""
    if yolo_model is None:
        return None, "YOLO model not loaded"
    
    try:
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            return None, "Could not read image"
        
        # Run YOLO detection
        results = yolo_model(img)[0]
        
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for i, box in enumerate(boxes):
                if confidences[i] >= YOLO_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box)
                    class_name = yolo_model.names[class_ids[i]]
                    confidence = confidences[i]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        return detections, None
    except Exception as e:
        return None, str(e)

# -----------------------------
# Draw bounding boxes on image
def draw_detections(img_path, detections, output_path):
    """Draw YOLO detections on image"""
    try:
        img = cv2.imread(img_path)
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Choose color based on confidence
            if confidence >= 0.80:
                color = (0, 255, 0)   # Green - high confidence
            elif confidence >= 0.60:
                color = (0, 255, 255) # Yellow - medium confidence
            else:
                color = (0, 165, 255) # Orange - low confidence
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name} {confidence*100:.1f}%"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        cv2.imwrite(output_path, img)
        return True
    except Exception as e:
        print(f"Error drawing detections: {e}")
        return False
def preprocess_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError("Could not load image")
    img_rgb = cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    img_input = preprocess_input(img_rgb.astype("float32"))
    img_input = np.expand_dims(img_input, axis=0)
    return img_rgb, img_input

# -----------------------------
# Log predictions
def log_prediction(class_label):
    stats[class_label] = stats.get(class_label, 0) + 1
    total_items = sum(stats.values())
    try:
        with open("waste_log.csv", "w") as f:
            f.write("Waste Classification Report\n")
            f.write(f"Total Items Processed: {total_items}\n")
            for category, count in stats.items():
                f.write(f"{category}: {count}\n")
    except Exception as e:
        print(f"Error writing log: {e}")

# -----------------------------
# Generate PDF report
def generate_pdf_report():
    try:
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
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None

# -----------------------------
# Image classification route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        detection_mode = request.form.get("detection_mode", "classification")
        
        if not file or file.filename == "":
            return redirect(request.url)

        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_extension not in allowed_extensions:
            return render_template("index.html", error="Please upload a valid image file (PNG, JPG, JPEG, GIF, BMP)")

        try:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            if detection_mode == "yolo" and yolo_model is not None:
                # YOLO Object Detection Mode
                detections, error = detect_objects_yolo(file_path)
                
                if error:
                    return render_template("index.html", error=f"YOLO detection error: {error}")
                
                if detections:
                    # Draw detections on image
                    output_filename = f"detected_{file.filename}"
                    output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
                    draw_detections(file_path, detections, output_path)
                    
                    # Log detections
                    for detection in detections:
                        log_prediction(detection['class'])
                    
                    return render_template(
                        "yolo_result.html",
                        original_image=file.filename,
                        detected_image=output_filename,
                        detections=detections,
                        detection_count=len(detections)
                    )
                else:
                    return render_template(
                        "yolo_result.html",
                        original_image=file.filename,
                        detected_image=file.filename,
                        detections=[],
                        detection_count=0,
                        message="No objects detected with sufficient confidence."
                    )
            
            else:
                # EfficientNet Classification Mode
                if best_model is None:
                    return render_template("index.html", error="Classification model not loaded. Please check if the model file exists.")

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
                
        except Exception as e:
            return render_template("index.html", error=f"Error processing image: {str(e)}")

    return render_template("index.html")

# -----------------------------
# Show statistics
@app.route("/stats")
def show_stats():
    if stats:
        try:
            categories = list(stats.keys())
            counts = list(stats.values())
            plt.figure(figsize=(10, 6))
            plt.bar(categories, counts, color="green", alpha=0.7)
            plt.xlabel("Category")
            plt.ylabel("Count")
            plt.title("Waste Classification Statistics")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Ensure static directory exists
            os.makedirs("static", exist_ok=True)
            plt.savefig("static/stats_chart.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating chart: {e}")
    
    return render_template("report.html", stats=stats)

# -----------------------------
# Download reports
@app.route("/download_pdf")
def download_pdf():
    try:
        pdf_path = generate_pdf_report()
        if pdf_path and os.path.exists(pdf_path):
            return send_file(pdf_path, as_attachment=True)
        else:
            return "Error generating PDF report", 500
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route("/download_csv")
def download_csv():
    try:
        if os.path.exists("waste_log.csv"):
            return send_file("waste_log.csv", as_attachment=True)
        else:
            return "No data to download", 404
    except Exception as e:
        return f"Error: {str(e)}", 500

# -----------------------------
# Real-time camera detection (disabled for Hugging Face Spaces)
@app.route("/camera")
def camera():
    return render_template("camera_disabled.html")

# Health check endpoint
@app.route("/health")
def health():
    return {"status": "healthy", "model_loaded": best_model is not None}

# Run Flask
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Hugging Face Spaces uses port 7860
    app.run(host="0.0.0.0", port=port, debug=False)