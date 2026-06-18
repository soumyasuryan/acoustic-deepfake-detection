import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from src.model import DeepfakeAcousticCNN, SemanticSpoofClassifier
from src.utils import preprocess_audio_gpu, get_text_embedding

class MultimodalDeepfakeDetector:
    def __init__(self, acoustic_weights_path="models/deepfake_cnn_weights.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Multimodal Engine on target device: {self.device.type.upper()}")
        
        self.transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0 if torch.cuda.is_available() else -1)
        self.nlp_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.nlp_model = AutoModel.from_pretrained("distilbert-base-uncased").to(self.device)
        self.nlp_model.eval()
        
        self.acoustic_model = DeepfakeAcousticCNN().to(self.device)
        if os.path.exists(acoustic_weights_path):
            self.acoustic_model.load_state_dict(torch.load(acoustic_weights_path, map_location=self.device))
            print(f"🎯 Success: Resumed optimized acoustic weights from '{acoustic_weights_path}'")
        else:
            print(f"⚠️ Warning: Pretrained acoustic weights not found at '{acoustic_weights_path}'. Using baseline.")
        self.acoustic_model.eval()
        
        self.semantic_model = SemanticSpoofClassifier().to(self.device)
        self.semantic_model.eval()

if __name__ == "__main__":
    detector = MultimodalDeepfakeDetector()
