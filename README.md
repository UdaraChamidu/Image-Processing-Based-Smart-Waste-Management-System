# Image-Processing-Based Smart Waste Management System

A deep learning-powered waste classification system that uses Convolutional Neural Networks (CNN) to automatically detect and categorize different types of solid waste materials. This system leverages computer vision and machine learning to enable smart waste management solutions.

## 🌟 Features

- **Automated Waste Classification**: Classifies waste into 6 categories (cardboard, glass, metal, paper, plastic, trash)
- **Advanced Image Processing**: Implements histogram equalization, Gaussian blur, and Sobel edge detection for enhanced image quality
- **Transfer Learning**: Utilizes MobileNetV2 pre-trained model for efficient and accurate classification
- **Data Augmentation**: Includes rotation, zoom, and flip transformations to improve model robustness
- **Real-time Prediction**: Supports single image prediction for practical applications
- **Comprehensive Evaluation**: Provides detailed performance metrics including confusion matrix and classification reports

## 🛠️ Technology Stack

- **Python 3.7+**
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision and image processing
- **scikit-learn**: Machine learning utilities
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Pandas**: Data manipulation
- **PIL**: Image processing

## 📁 Dataset

This project uses the **TrashNet Dataset** from Hugging Face:
- **Source**: [garythung/trashnet](https://huggingface.co/datasets/garythung/trashnet)
- **Categories**: 6 waste types (cardboard, glass, metal, paper, plastic, trash)
- **Images**: High-resolution images of various waste materials
- **Format**: RGB images resized to 224x224 for MobileNetV2 compatibility

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Image-Processing-Based-Smart-Waste-Management-System.git
cd Image-Processing-Based-Smart-Waste-Management-System
```

2. **Install required packages**:
```bash
pip install datasets tensorflow opencv-python matplotlib pandas scikit-learn pillow seaborn
```

3. **Install Git LFS** (for large files):
```bash
git lfs install
```

## 📊 Model Architecture

### Base Model: MobileNetV2
- **Pre-trained weights**: ImageNet
- **Input shape**: (224, 224, 3)
- **Feature extraction**: Frozen base layers
- **Custom layers**:
  - Global Average Pooling 2D
  - Dropout (0.5)
  - Dense layer (6 classes, softmax activation)

### Image Enhancement Pipeline
1. **Resize**: 128x128 → 224x224
2. **Grayscale conversion**
3. **Histogram equalization** for contrast enhancement
4. **Gaussian blur** for noise reduction
5. **Sobel edge detection** for feature enhancement
6. **Normalization** to [0,1] range

## 🏃‍♂️ Usage

### Training the Model

```python
# The main training script includes:
# 1. Data loading and preprocessing
# 2. Image enhancement
# 3. Model creation and compilation
# 4. Training with callbacks
# 5. Model evaluation

# Run the complete training pipeline:
python solid_waste_detection_cnn.py
```

### Making Predictions

```python
from PIL import Image
import numpy as np

def predict_image(path):
    img = Image.open(path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    predicted_class = le.inverse_transform([class_idx])[0]
    confidence = np.max(pred) * 100
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    return predicted_class, confidence

# Example usage
predict_image("path/to/your/waste/image.jpg")
```

## 📈 Model Performance

### Training Configuration
- **Optimizer**: Adam (learning rate: 1e-3)
- **Loss function**: Categorical crossentropy
- **Batch size**: 32
- **Epochs**: 30 (with early stopping)
- **Train/Test split**: 80/20

### Callbacks Used
- **Early Stopping**: Prevents overfitting (patience=5)
- **Learning Rate Reduction**: Adaptive learning (patience=3, factor=0.2)
- **Model Checkpoint**: Saves best performing model

### Data Augmentation
- Rotation range: 30°
- Zoom range: 0.2
- Width/Height shift: 0.1
- Horizontal flip: Enabled

## 🔍 Evaluation Metrics

The model provides comprehensive evaluation through:

1. **Classification Report**: Precision, recall, and F1-score for each class
2. **Confusion Matrix**: Visual representation of classification performance
3. **Training Curves**: Accuracy and loss plots for training/validation
4. **Per-class Analysis**: Individual performance metrics

## 📂 Project Structure

```
Image-Processing-Based-Smart-Waste-Management-System/
│
├── solid_waste_detection_cnn.py    # Main training script
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── models/                         # Saved models directory
│   ├── best_model.keras           # Best performing model
│   └── trashnet_mobilenetv2_model.h5  # Final trained model
├── data/                          # Dataset directory
│   └── dataset-original/          # TrashNet dataset
├── examples/                      # Example images and predictions
└── notebooks/                     # Jupyter notebooks (if any)
```

## 🎯 Applications

This system can be integrated into various smart waste management solutions:

- **Smart Bins**: Automatic waste sorting at collection points
- **Recycling Plants**: Automated material separation
- **Mobile Apps**: User-friendly waste classification tools
- **IoT Systems**: Integration with smart city infrastructure
- **Educational Tools**: Environmental awareness applications

## 🤝 Contributing

Contributions are welcome! Here are ways you can contribute:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Improvement
- Add more waste categories
- Implement real-time video processing
- Create mobile app interface
- Add multi-language support
- Optimize model for edge devices

## 📋 Requirements

Create a `requirements.txt` file with:

```
tensorflow>=2.8.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.3.0
numpy>=1.21.0
seaborn>=0.11.0
pillow>=8.0.0
datasets>=2.0.0
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TrashNet Dataset**: Thanks to Gary Thung for providing the comprehensive waste classification dataset
- **MobileNetV2**: Google's efficient neural network architecture
- **TensorFlow Team**: For the excellent deep learning framework
- **OpenCV Community**: For computer vision tools and libraries

## 📞 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **Project Link**: [https://github.com/yourusername/Image-Processing-Based-Smart-Waste-Management-System](https://github.com/yourusername/Image-Processing-Based-Smart-Waste-Management-System)

## 🔮 Future Enhancements

- [ ] Real-time waste detection using webcam
- [ ] Mobile application development
- [ ] Integration with IoT sensors
- [ ] Multi-modal classification (text + image)
- [ ] Deployment on edge devices
- [ ] API development for third-party integration
- [ ] Support for additional waste categories
- [ ] Contamination detection in recyclables

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
