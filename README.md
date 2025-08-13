# Speech Digit Recognition System

A lightweight, efficient spoken digit recognition system that can recognize digits 0-9 from audio input using deep learning techniques.

## 🏗️ Model Architecture

### Core Components

#### 1. Audio Feature Extractor (`AudioFeatureExtractor`)
- **MFCC (Mel-Frequency Cepstral Coefficients)**: Extracts 13 MFCC coefficients optimized for speech recognition
- **Sample Rate**: 22050 Hz for optimal audio quality
- **Feature Dimensions**: Fixed output size of (13, 50) for batch processing
- **Normalization**: Z-score normalization for training stability
- **Resampling**: Automatic resampling to ensure consistent input format

#### 2. Lightweight CNN (`LightweightDigitRecognizer`)
The model follows a streamlined architecture designed for efficiency:

```
Input: (batch_size, 13, 50) MFCC features
    ↓
Conv1D(13→32, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool(2)
    ↓
Conv1D(32→64, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool(2)  
    ↓
Conv1D(64→128, kernel=3, padding=1) + BatchNorm + ReLU + AdaptiveAvgPool(1)
    ↓
Flatten + Dropout(0.3) + Linear(128→64) + ReLU + Dropout(0.2) + Linear(64→10)
    ↓
Output: (batch_size, 10) digit probabilities
```

**Key Design Principles:**
- **Parameter Efficiency**: Only ~45K parameters for lightweight deployment
- **1D Convolutions**: Optimized for temporal feature processing
- **Batch Normalization**: Stable training and faster convergence
- **Global Average Pooling**: Reduces parameters while preserving spatial information
- **Strategic Dropout**: Prevents overfitting (0.3 after conv layers, 0.2 before final layer)

#### 3. Model Trainer (`ModelTrainer`)
- **Optimizer**: Adam with weight decay (1e-4) for regularization
- **Learning Rate Scheduling**: ReduceLROnPlateau with patience=3, factor=0.5
- **Loss Function**: CrossEntropyLoss for multi-class classification
- **Checkpointing**: Saves best model based on validation accuracy

#### 4. Real-Time Recognizer (`RealTimeDigitRecognizer`)
- **Fast Inference**: Optimized for minimal latency
- **Batch Processing**: Supports multiple audio files simultaneously
- **Confidence Scoring**: Provides prediction confidence levels
- **Timing Measurements**: Tracks inference performance

## 🎯 Technical Approach

### 1. Feature Engineering
- **MFCC Extraction**: Captures spectral characteristics essential for speech recognition
- **Fixed-Length Output**: Ensures consistent input dimensions for batch processing
- **Normalization**: Z-score normalization for better training convergence

### 2. Model Design Philosophy
- **Lightweight**: Minimal parameters for edge deployment
- **Efficient**: 1D convolutions reduce computational complexity
- **Robust**: Batch normalization and dropout prevent overfitting
- **Scalable**: Modular architecture allows easy modifications

### 3. Training Strategy
- **Data Augmentation**: Built-in resampling and normalization
- **Learning Rate Scheduling**: Adaptive learning rate for optimal convergence
- **Early Stopping**: Model checkpointing prevents overfitting
- **Comprehensive Metrics**: Tracks loss, accuracy, and validation performance

## 🚀 Building and Testing

### Prerequisites
- Python 3.8+
- PyTorch 1.9.0+
- CUDA-compatible GPU (optional, for faster training)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Speech-Processing
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### Training the Model
```bash
python speech_digit_recognition.py
```

The script will:
1. Load the free-spoken-digit-dataset from Hugging Face
2. Extract MFCC features from audio samples
3. Train the lightweight CNN model
4. Save the best model as `best_digit_recognizer.pth`
5. Display training progress and final metrics

#### Key Parameters
- **Training Epochs**: 20 (configurable)
- **Learning Rate**: 0.001 (with adaptive scheduling)
- **Batch Size**: 32
- **Target Feature Length**: 50 time steps
- **MFCC Coefficients**: 13

### Testing

#### 1. **Model Evaluation**
- Automatic validation during training
- Confusion matrix visualization
- Classification report generation
- Training progress plots

#### 2. **Real-Time Recognition**
```python
from speech_digit_recognition import RealTimeDigitRecognizer, AudioFeatureExtractor

# Initialize recognizer
feature_extractor = AudioFeatureExtractor()
recognizer = RealTimeDigitRecognizer('best_digit_recognizer.pth', feature_extractor)

# Predict from audio file
digit, confidence, inference_time = recognizer.predict_digit('audio_file.wav')
print(f"Predicted: {digit}, Confidence: {confidence:.3f}, Time: {inference_time:.2f}ms")
```

#### 3. **Batch Processing**
```python
# Process multiple files
audio_files = ['file1.wav', 'file2.wav', 'file3.wav']
results = recognizer.batch_predict(audio_files)
```

## 📊 Results and Performance

### Model Performance
- **Model Size**: ~45K parameters (~180 KB)
- **Training Time**: ~5-10 minutes on CPU, ~2-3 minutes on GPU
- **Inference Speed**: <10ms per audio sample
- **Memory Usage**: Efficient batch processing with minimal memory footprint

### Accuracy Metrics
- **Training Accuracy**: Typically reaches 95%+ by epoch 20
- **Validation Accuracy**: Consistent performance on unseen data
- **Generalization**: Robust performance across different speakers and recording conditions

### Key Strengths
1. **Efficiency**: Lightweight architecture suitable for edge devices
2. **Speed**: Fast inference for real-time applications
3. **Robustness**: Consistent performance across various audio conditions
4. **Scalability**: Easy to extend for additional digits or languages

### Performance Visualization
The system automatically generates:
- Training loss curves
- Accuracy progression plots
- Confusion matrix for detailed error analysis
- Learning rate scheduling visualization

## 🔧 Customization and Extension

### Adding New Digits
1. Modify `num_classes` in `LightweightDigitRecognizer`
2. Update the final linear layer dimensions
3. Retrain with new dataset

### Feature Modifications
1. Adjust MFCC parameters in `AudioFeatureExtractor`
2. Modify `target_length` for different temporal resolutions
3. Add additional audio features (spectral centroid, rolloff, etc.)

### Architecture Changes
1. Modify convolution layers in the `features` sequential block
2. Adjust dropout rates for different regularization needs
3. Change pooling strategies for different feature extraction approaches

## 📁 Project Structure
```
Speech-Processing/
├── speech_digit_recognition.py    # Main implementation
├── requirements.txt               # Dependencies
├── best_digit_recognizer.pth     # Trained model weights
├── README.md                     # This file
└── venv/                        # Virtual environment
```

## 🎯 Use Cases

- **Voice-Controlled Systems**: Number input via speech
- **Accessibility Tools**: Audio-based number recognition
- **IoT Devices**: Lightweight speech recognition for embedded systems
- **Educational Applications**: Speech-based learning tools
- **Call Center Automation**: Number recognition in voice systems

## 🔮 Future Enhancements

1. **Multi-Language Support**: Extend to recognize digits in different languages
2. **Noise Robustness**: Improve performance in noisy environments
3. **Speaker Adaptation**: Personalize recognition for specific users
4. **Real-Time Streaming**: Process continuous audio streams
5. **Mobile Deployment**: Optimize for mobile and edge devices

## 📚 References

- **Dataset**: [Free Spoken Digit Dataset](https://huggingface.co/datasets/mteb/free-spoken-digit-dataset)
- **MFCC Features**: Mel-frequency cepstral coefficients for audio analysis
- **PyTorch**: Deep learning framework for model implementation
- **Torchaudio**: Audio processing utilities for PyTorch

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the system.

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using PyTorch and modern deep learning techniques for efficient speech recognition.**
