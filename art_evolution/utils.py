import sys
import time
import io
import requests
import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple, Any, Set
import os

from model_loaders import get_aesthetic_model, get_resnet_model

def generate_artwork(
    prompt: str,
    name: str,
    output_dir: str,
    client,
    model="dall-e-3",
    size="1024x1024",
    quality="standard",
    style="vivid",
    suffix="",
):
    """Generate artwork from prompt using DALL-E or other image generation models"""
    
    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1,
        )

        # Save the image
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        image = Image.open(io.BytesIO(image_response.content))
        
        # Create filename from name
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
        filename = f"{safe_name}{suffix}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        image.save(filepath)
        
        return filepath
    except Exception as e:
        sys.stderr.write(f"Error generating artwork: {e}\n")
        return None

def evaluate_aesthetic_score(artwork_path):
    """Evaluate the generated artwork using aesthetic predictor"""
    try:
        # Get cached model and preprocessor
        model, preprocessor = get_aesthetic_model()
        
        # Load image to evaluate
        image = Image.open(artwork_path).convert("RGB")

        # Preprocess image
        pixel_values = (
            preprocessor(images=image, return_tensors="pt")
            .pixel_values
        )
        
        # Move to appropriate device - get device from model
        device = next(model.parameters()).device
        pixel_values = pixel_values.to(device)

        # Predict aesthetic score
        with torch.inference_mode():
            score = model(pixel_values).logits.squeeze().float().cpu().numpy()
        
        # Return success and the score
        return True, float(score)
    
    except Exception as e:
        sys.stderr.write(f"Error evaluating artwork aesthetic score: {e}\n")
        return False, str(e)

def evaluate_concept_diversity(concepts, concept_model):
    """Evaluate the diversity of concepts used in the generation based on cosine similarity"""
    try:
        if not concept_model or len(concepts) < 2:
            return True, 0.0  # Default value changed from 0.5 to 0.0
            
        # Calculate similarity for all pairs
        similarities = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                similarity = concept_model.get_similarity(concepts[i], concepts[j])
                similarities.append(similarity)
                
        # Calculate average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Convert to diversity score (lower similarity = higher diversity)
        diversity_score = 1.0 - avg_similarity
        
        return True, float(diversity_score)
    
    except Exception as e:
        sys.stderr.write(f"Error evaluating concept diversity: {e}\n")
        return False, str(e)

def evaluate_originality(artwork_path):
    """Evaluate the cosine similarity between the generated artwork and WikiArt embeddings"""
    try:
        # Get ResNet model and transform
        model, transform = get_resnet_model()
        
        # Load image
        image = Image.open(artwork_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Move to appropriate device - get device from model
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Get embedding for the generated image
        with torch.no_grad():
            embedding = model(image_tensor).squeeze().cpu().numpy()
        
        # Load WikiArt embeddings
        data = np.load('data/embedding/embeddings.npy', allow_pickle=True).item()
        wikiart_embeddings = data['embeddings']
        
        # Calculate cosine similarity with all WikiArt embeddings
        similarities = cosine_similarity([embedding], wikiart_embeddings)[0]
        
        # Find the maximum similarity (closest match)
        max_similarity = np.max(similarities)
                
        return True, float(max_similarity)
    
    except Exception as e:
        sys.stderr.write(f"Error evaluating artwork originality: {e}\n")
        return False, str(e)

def evaluate_combined(artwork_path, concepts=None, concept_model=None):
    """Evaluate artwork using aesthetic score, cosine similarity, and concept diversity"""
    # Get aesthetic score
    aesthetic_success, aesthetic_score = evaluate_aesthetic_score(artwork_path)
    if not aesthetic_success:
        return False, aesthetic_score
    
    # Get cosine similarity for originality
    similarity_success, similarity_score = evaluate_originality(artwork_path)
    if not similarity_success:
        return False, similarity_score
    
    # Get concept diversity if concepts and concept model provided
    diversity_score = 0.0
    if concepts and concept_model and len(concepts) >= 2:
        diversity_success, diversity_score = evaluate_concept_diversity(concepts, concept_model)
        if not diversity_success:
            # Use default value if failed
            diversity_score = 0.0
    
    # Normalize aesthetic score from 0-10 range to 0-1 range
    normalized_aesthetic = aesthetic_score / 10.0
    
    # Invert similarity for originality
    originality_score = 1.0 - similarity_score
    
    # Combine normalized scores (equal weighting of all three factors)
    combined_score = (normalized_aesthetic + originality_score + diversity_score) / 3.0
        
    return True, (combined_score, aesthetic_score, originality_score, diversity_score)

def get_fitness_function(fitness_type="combined"):
    """Return the appropriate fitness function based on the specified type.
    
    Args:
        fitness_type (str): Type of fitness function to use. 
                          Options: "combined", "aesthetic", "originality", "diversity" 
    
    Returns:
        function: The selected fitness function
    """
    if fitness_type == "aesthetic":
        # Wrapper for aesthetic-only fitness
        def evaluate_aesthetic_only(artwork_path, concepts=None, concept_model=None):
            success, score = evaluate_aesthetic_score(artwork_path)
            if not success:
                return False, score
            
            # Normalize to 0-1 range and return in the same format as evaluate_combined
            normalized_score = score / 10.0
            return True, (normalized_score, score, 0.0, 0.0)  # Main score, aesthetic, originality (0), diversity (0)
        
        return evaluate_aesthetic_only
    
    elif fitness_type == "originality":
        # Wrapper for originality-only fitness
        def evaluate_originality_only(artwork_path, concepts=None, concept_model=None):
            success, similarity = evaluate_originality(artwork_path)
            if not success:
                return False, similarity
            
            # Convert similarity to originality score (1 - similarity)
            originality = 1.0 - similarity
            
            # Return in the same format as evaluate_combined
            return True, (originality, 0.0, originality, 0.0)  # Main score, aesthetic (0), originality, diversity (0)
        
        return evaluate_originality_only
    
    elif fitness_type == "diversity":
        # Wrapper for diversity-only fitness
        def evaluate_diversity_only(artwork_path, concepts=None, concept_model=None):
            if not concepts or not concept_model or len(concepts) < 2:
                return True, (0.0, 0.0, 0.0, 0.0)  # Default values changed from 0.5 to 0.0
            
            success, diversity = evaluate_concept_diversity(concepts, concept_model)
            if not success:
                return False, diversity
            
            # Return in the same format as evaluate_combined
            return True, (diversity, 0.0, 0.0, diversity)  # Main score, aesthetic (0), originality (0), diversity
        
        return evaluate_diversity_only
    
    else:  # Default to combined
        return evaluate_combined

def validate_prompt(prompt_data, original_concepts: List[str]):
    """Validate that the prompt data contains required fields and includes original concepts"""
    from concept_management import Action
    
    required_fields = ["thought", "name", "prompt", "action", "concepts_used"]
    
    # Check basic required fields
    for field in required_fields:
        if field not in prompt_data:
            return False, f"Missing required field: {field}"
    
    # Validate action
    try:
        action = Action(prompt_data["action"])
    except ValueError:
        return False, f"Invalid action: {prompt_data['action']}. Must be one of: {[a.value for a in Action]}"
    
    # Validate that all original concepts are included in concepts_used
    concepts_used = set(prompt_data["concepts_used"])
    missing_original = [c for c in original_concepts if c not in concepts_used]
    if missing_original:
        return False, f"Missing original concepts in concepts_used: {missing_original}"
    
    # Validate action-specific fields
    if action == Action.ADD_ONE_CONCEPT:
        if "new_concept" not in prompt_data:
            return False, "Missing new_concept field for ADD_ONE_CONCEPT action"
        if not prompt_data["new_concept"]:
            return False, "new_concept cannot be empty"
    elif action == Action.ADD_MULTIPLE_CONCEPTS:
        if "new_concepts" not in prompt_data:
            return False, "Missing new_concepts field for ADD_MULTIPLE_CONCEPTS action"
        if not prompt_data["new_concepts"] or not isinstance(prompt_data["new_concepts"], list):
            return False, "new_concepts must be a non-empty list for ADD_MULTIPLE_CONCEPTS action"
    
    # Validate that concepts_used contains at least the original concepts
    if len(prompt_data["concepts_used"]) < len(original_concepts):
        return False, "concepts_used must include at least all original concepts for recombination"
    
    return True, "" 