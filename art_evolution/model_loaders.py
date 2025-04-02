import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet152_Weights
import clip
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

# Global variables for model caching
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None
_AESTHETIC_MODEL = None
_AESTHETIC_PREPROCESSOR = None
_RESNET_MODEL = None
_RESNET_TRANSFORM = None

def get_clip_model():
    """Get cached CLIP model or load it if not available"""
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    if _CLIP_MODEL is None or _CLIP_PREPROCESS is None:
        sys.stderr.write("Loading CLIP model (first time only)...\n")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP_DEVICE = device
        
        if device == "cpu":
            sys.stderr.write("Warning: Running CLIP model on CPU. This will be significantly slower.\n")
            
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=device)
        sys.stderr.write(f"CLIP model loaded successfully on {device}\n")
    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE

def get_aesthetic_model():
    """Get cached aesthetic prediction model or load it if not available"""
    global _AESTHETIC_MODEL, _AESTHETIC_PREPROCESSOR
    if _AESTHETIC_MODEL is None or _AESTHETIC_PREPROCESSOR is None:
        sys.stderr.write("Loading aesthetic prediction model (first time only)...\n")
        _AESTHETIC_MODEL, _AESTHETIC_PREPROCESSOR = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        # Check if CUDA is available before using it
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            _AESTHETIC_MODEL = _AESTHETIC_MODEL.to(torch.bfloat16).cuda()
        else:
            # Use CPU with appropriate dtype
            _AESTHETIC_MODEL = _AESTHETIC_MODEL.to(torch.float32)
            sys.stderr.write("Warning: Running aesthetic model on CPU. This will be significantly slower.\n")
            
        sys.stderr.write(f"Aesthetic model loaded successfully on {device}\n")
    return _AESTHETIC_MODEL, _AESTHETIC_PREPROCESSOR

def get_resnet_model():
    """Get cached ResNet model or load it if not available"""
    global _RESNET_MODEL, _RESNET_TRANSFORM
    if _RESNET_MODEL is None or _RESNET_TRANSFORM is None:
        sys.stderr.write("Loading ResNet model for embeddings (first time only)...\n")
        
        # Load the ResNet152 model with the latest weights
        weights = ResNet152_Weights.DEFAULT
        model = models.resnet152(weights=weights)
        num_ftrs = model.fc.in_features
        
        # Modify the final fully connected layer
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 26)
        )
        
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the saved model weights with appropriate device mapping
        model_path = 'data/embedding/best_model_resnet_152.pth'
        
        if device.type == 'cuda':
            model.load_state_dict(torch.load(model_path))
        else:
            # Load weights on CPU when CUDA is not available
            model.load_state_dict(torch.load(model_path, map_location=device))
            sys.stderr.write("Warning: Running ResNet model on CPU. This will be significantly slower.\n")
            
        model = model.to(device)
        model.eval()
        
        # Extract feature extractor (everything except the classification layer)
        _RESNET_MODEL = nn.Sequential(*list(model.children())[:-1])
        
        # Define the transformation for ResNet
        _RESNET_TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        sys.stderr.write(f"ResNet model loaded successfully on {device}\n")
    return _RESNET_MODEL, _RESNET_TRANSFORM 