import pyaudio
import numpy as np
import torch
import torchaudio
import librosa
from io import BytesIO
import warnings
import time
import os

# Suppress PyAudio and other warnings
warnings.filterwarnings("ignore")
# This class defines the model architecture.
class LightweightDigitRecognizer(torch.nn.Module):
    def __init__(self, num_classes: int = 10, input_channels: int = 13):
        super(LightweightDigitRecognizer, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, num_classes)
        )
        # Initialize weights with a basic method
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
# This class defines the audio preprocessing and feature extraction logic.
class AudioFeatureExtractor:
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13, 
                 n_fft: int = 2048, hop_length: int = 512, target_length: int = 50):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = target_length
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
            }
        )

    def extract_features(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Extract MFCC features from a NumPy waveform array.
        
        Args:
            waveform: Raw audio data as a NumPy array.
            
        Returns:
            Normalized MFCC features as a PyTorch tensor.
        """
        try:
            # Convert NumPy array to a PyTorch tensor
            waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)

            # Apply the MFCC transform
            mfcc = self.mfcc_transform(waveform_tensor)

            # Pad or truncate to fixed length
            if mfcc.shape[2] > self.target_length:
                mfcc = mfcc[:, :, :self.target_length]
            else:
                padding = self.target_length - mfcc.shape[2]
                mfcc = torch.nn.functional.pad(mfcc, (0, padding))

            # Normalize features
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
            
            return mfcc.squeeze(0) # Squeeze to remove the channel dimension

        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

def get_real_time_prediction(model, feature_extractor, audio_stream, confidence_threshold=0.6):
    """
    Captures audio from the microphone, processes it, and makes a prediction.
    Only returns a prediction if the confidence is above the threshold.
    """
    print("\nListening for a spoken digit... (Speak for ~1 second)")
    
    # Capture audio data from the stream
    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)):
        data = audio_stream.read(CHUNK)
        frames.append(data)
    
    print("Recording finished. Processing...")

    # Concatenate the frames and convert to NumPy array
    audio_data = b''.join(frames)
    # The format 'paInt16' corresponds to int16 data type
    waveform_np = np.frombuffer(audio_data, dtype=np.int16)

    # Normalize waveform to float values
    waveform_np = waveform_np.astype(np.float32) / 32768.0

    # Extract features
    features = feature_extractor.extract_features(waveform_np)
    if features is None:
        print("Failed to extract features.")
        return None, None

    # Make a prediction
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        features_batch = features.unsqueeze(0)
        outputs = model(features_batch)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_digit = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_digit].item()
    
    # Check if the confidence is above the threshold
    if confidence > confidence_threshold:
        return predicted_digit, confidence
    else:
        return None, None

if __name__ == "__main__":
    # --- 3. Setup and Execution ---
    
    # Audio configuration for PyAudio
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = 22050
    RECORD_SECONDS = 1.5 # Record for 1.5 seconds

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Load model and feature extractor
    feature_extractor = AudioFeatureExtractor(sample_rate=SAMPLE_RATE, n_mfcc=13)
    # Open the audio stream
    audio_stream = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=SAMPLE_RATE,
                          input=True,
                          frames_per_buffer=CHUNK)

    MODEL_FILE_PATH = 'best_digit_recognizer.pth'
    if os.path.exists(MODEL_FILE_PATH):
        print(f"Loading trained model from {MODEL_FILE_PATH}...")
        try:
            model = LightweightDigitRecognizer(input_channels=13)
            model.load_state_dict(torch.load(MODEL_FILE_PATH))
            print("Trained model loaded successfully.")
        except Exception as e:
            print(f"Error loading trained model: {e}")
            print("Falling back to a dummy model for demonstration.")
            model = LightweightDigitRecognizer(input_channels=13)

    print("PyAudio is ready. Press Ctrl+C to stop.")

    try:
        while True:
            # Get a real-time prediction
            predicted_digit, confidence = get_real_time_prediction(model, feature_extractor, audio_stream)

            if predicted_digit is not None:
                print(f"Predicted: {predicted_digit} (Confidence: {confidence:.2f})\n")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Close and terminate the audio stream
        audio_stream.stop_stream()
        audio_stream.close()
        p.terminate()

