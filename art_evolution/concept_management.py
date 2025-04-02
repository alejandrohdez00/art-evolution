import sys
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from model_loaders import get_clip_model
from matplotlib.patches import Patch

class Action(Enum):
    ADD_NO_CONCEPT = "add_no_concept"
    ADD_ONE_CONCEPT = "add_one_concept"
    ADD_MULTIPLE_CONCEPTS = "add_multiple_concepts"
    RECOMBINATION = "recombination"  # Keep as an enum even though it's mandatory

class ConceptEmbedding:
    def __init__(self, concept: str, embedding: np.ndarray):
        self.concept = concept
        self.embedding = embedding
        self.creation_time = time.time()
        self.usage_count = 0
        self.unsuccessful_uses = 0  # Track unsuccessful uses
        self.is_expired = False  # Track if concept is expired
        self.expired_at_generation = None  # Track at which generation the concept expired
        
    def update_usage(self, improved_fitness: bool = False, is_original: bool = False):
        """Update usage count and track if usage improved fitness"""
        self.usage_count += 1
        # Only track unsuccessful uses for non-original concepts
        if not is_original:
            if not improved_fitness:
                self.unsuccessful_uses += 1
            else:
                # Reset unsuccessful uses counter when concept contributes to fitness improvement
                self.unsuccessful_uses = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "concept": self.concept,
            "embedding": self.embedding.astype(float).tolist(),
            "creation_time": self.creation_time,
            "usage_count": self.usage_count,
            "unsuccessful_uses": self.unsuccessful_uses,
            "is_expired": self.is_expired,
            "expired_at_generation": self.expired_at_generation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptEmbedding':
        """Create from dictionary"""
        embedding = np.array(data["embedding"])
        concept_embedding = cls(data["concept"], embedding)
        concept_embedding.creation_time = data["creation_time"]
        concept_embedding.usage_count = data["usage_count"]
        concept_embedding.unsuccessful_uses = data.get("unsuccessful_uses", 0)
        concept_embedding.is_expired = data.get("is_expired", False)
        concept_embedding.expired_at_generation = data.get("expired_at_generation", None)
        return concept_embedding

class ConceptSpaceModel:
    def __init__(self, client, embedding_model="openai"):
        self.client = client
        self.embedding_model = embedding_model
        self.concept_embeddings = {}  # concept -> ConceptEmbedding
        
    def add_concept(self, concept: str) -> None:
        """Generate and store embedding for a new concept"""
        if concept not in self.concept_embeddings:
            embedding = self._generate_embedding(concept)
            self.concept_embeddings[concept] = ConceptEmbedding(concept, embedding)
            
    def _generate_embedding(self, concept: str) -> np.ndarray:
        """Generate embedding for a concept using the selected embedding model"""
        try:
            if self.embedding_model == "clip":
                # Use CLIP for embeddings
                model, _, device = get_clip_model()
                with torch.no_grad():
                    text_inputs = clip.tokenize([concept]).to(device)
                    text_features = model.encode_text(text_inputs)
                    embedding = text_features.cpu().numpy()[0]
                    # Normalize the embedding
                    embedding = embedding / np.linalg.norm(embedding)
                return embedding
            else:
                # Use OpenAI's embedding API (default)
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=concept,
                    dimensions=1536
                )
                embedding = np.array(response.data[0].embedding)
                return embedding
        except Exception as e:
            sys.stderr.write(f"Error generating embedding for {concept}: {e}\n")
            # Raise the error instead of returning a random embedding
            raise ValueError(f"Failed to generate embedding for concept '{concept}': {e}")
    
    def get_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate cosine similarity between two concepts"""
        if concept1 not in self.concept_embeddings:
            self.add_concept(concept1)
        if concept2 not in self.concept_embeddings:
            self.add_concept(concept2)
            
        emb1 = self.concept_embeddings[concept1].embedding
        emb2 = self.concept_embeddings[concept2].embedding
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def calculate_novelty(self, new_concept: str, existing_concepts: List[str]) -> float:
        """Calculate how novel a concept is compared to existing concepts"""
        if not existing_concepts:
            return 1.0  # Maximum novelty if no existing concepts
            
        similarities = [self.get_similarity(new_concept, c) for c in existing_concepts]
        # Lower max similarity means higher novelty
        return 1.0 - max(similarities)
        
    def get_concept_clusters(self, concepts: List[str], n_clusters: int = 3) -> Dict[int, List[str]]:
        """Group concepts into semantic clusters"""
        if len(concepts) < n_clusters:
            return {0: concepts}  # Not enough concepts to cluster
            
        # Get embeddings for all concepts
        for concept in concepts:
            if concept not in self.concept_embeddings:
                self.add_concept(concept)
                
        embeddings = np.array([self.concept_embeddings[c].embedding for c in concepts])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(concepts)), random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Group concepts by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(concepts[i])
            
        return clusters
        
    def get_diverse_subset(self, concepts: List[str], n: int) -> List[str]:
        """Select a diverse subset of concepts"""
        if len(concepts) <= n:
            return concepts
            
        # Ensure all concepts have embeddings
        for concept in concepts:
            if concept not in self.concept_embeddings:
                self.add_concept(concept)
                
        embeddings = np.array([self.concept_embeddings[c].embedding for c in concepts])
        selected_indices = [0]  # Start with the first concept
        
        while len(selected_indices) < n:
            # Calculate distance from each unselected concept to all selected concepts
            unselected = [i for i in range(len(concepts)) if i not in selected_indices]
            min_distances = []
            
            for i in unselected:
                distances = [1.0 - cosine_similarity([embeddings[i]], [embeddings[j]])[0][0] for j in selected_indices]
                min_distances.append((i, min(distances)))  # Use minimum distance to any selected concept
                
            # Select the concept with maximum minimum distance (most different)
            next_concept = max(min_distances, key=lambda x: x[1])[0]
            selected_indices.append(next_concept)
            
        return [concepts[i] for i in selected_indices]
    
    def suggest_new_concept(self, existing_concepts: List[str], n_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Suggest new concepts that would be novel additions to the existing pool"""
        # TO-DO: This would typically use the LLM to generate suggestions or use the list of predefined concepts
        potential_concepts = [
            "cubism", "surrealism", "impressionism", "expressionism", 
            "minimalism", "abstract", "pop art", "futurism", "dadaism",
            "cyberpunk", "steampunk", "vaporwave", "baroque", "renaissance",
            "gothic", "art nouveau", "art deco", "photorealism", "hyperrealism",
            "pointillism", "fauvism", "constructivism", "suprematism", "neoclassicism"
        ]
        
        # Filter out concepts that are already in the pool
        new_concepts = [c for c in potential_concepts if c not in existing_concepts]
        
        # Calculate novelty scores for each potential new concept
        novelty_scores = [(c, self.calculate_novelty(c, existing_concepts)) for c in new_concepts]
        
        # Sort by novelty score and return top n
        return sorted(novelty_scores, key=lambda x: x[1], reverse=True)[:n_suggestions]
    
    def visualize_concept_space(self, concepts: List[str], save_path: str, concept_expired_status: Dict[str, bool]) -> None:
        """Visualize the concept space using t-SNE"""
        if len(concepts) < 2:
            sys.stderr.write("Need at least 2 concepts to visualize concept space\n")
            return
            
        # Ensure all concepts have embeddings
        for concept in concepts:
            if concept not in self.concept_embeddings:
                self.add_concept(concept)
                
        # Get embeddings and prepare for t-SNE
        embeddings = np.array([self.concept_embeddings[c].embedding for c in concepts])
        
        n_samples = len(concepts)
        if n_samples <= 5:
            # For very few samples, use a very small perplexity
            perplexity = max(2, n_samples - 1)
            sys.stderr.write(f"Warning: Few concepts ({n_samples}), using reduced perplexity of {perplexity}\n")
        else:
            # Otherwise use min(30, n_samples/3) as a rule of thumb
            perplexity = min(30, n_samples // 3)
        
        # Apply t-SNE for dimensionality reduction
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            reduced_embeddings = tsne.fit_transform(embeddings)
            
            # Create plot
            plt.figure(figsize=(14, 12))
            
            # Plot active and expired concepts with different colors
            active_concepts_list = [c for c in concepts if not concept_expired_status[c]]
            expired_concepts_list = [c for c in concepts if concept_expired_status[c]]
            
            # Get all concept coordinates
            all_indices = range(len(concepts))
            all_x = reduced_embeddings[all_indices, 0]
            all_y = reduced_embeddings[all_indices, 1]
            
            # Plot all points with small transparent markers just to establish the space
            plt.scatter(all_x, all_y, s=50, alpha=0.0)
            
            # Add labels with colored backgrounds for each point
            for i, concept in enumerate(concepts):
                is_expired = concept_expired_status[concept]
                bg_color = 'orange' if is_expired else 'lightgray'
                text_color = 'black'
                plt.annotate(concept, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), 
                            fontsize=14, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.6", fc=bg_color, ec="gray", alpha=0.95, linewidth=2))
                   
            legend_elements = [
                Patch(facecolor='lightgray', edgecolor='gray', label='Active Concepts'),
                Patch(facecolor='orange', edgecolor='gray', label='Expired Concepts')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.title("Concept Space Visualization", fontsize=18)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the visualization
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            sys.stderr.write(f"Concept space visualization saved to: {save_path}\n")
        except Exception as e:
            sys.stderr.write(f"Error in t-SNE visualization: {e}\n")
            
            # If t-SNE fails, create a simple 2D plot using PCA as fallback
            if len(concepts) >= 2:
                try:
                    pca = PCA(n_components=2)
                    reduced_embeddings = pca.fit_transform(embeddings)
                    
                    plt.figure(figsize=(14, 12))
                    
                    # Plot active and expired concepts with different colors
                    active_concepts_list = [c for c in concepts if not concept_expired_status[c]]
                    expired_concepts_list = [c for c in concepts if concept_expired_status[c]]
                    
                    # Get all concept coordinates
                    all_indices = range(len(concepts))
                    all_x = reduced_embeddings[all_indices, 0]
                    all_y = reduced_embeddings[all_indices, 1]
                    
                    # Plot all points with small transparent markers just to establish the space
                    plt.scatter(all_x, all_y, s=50, alpha=0.0)
                    
                    # Add labels with colored backgrounds for each point
                    for i, concept in enumerate(concepts):
                        is_expired = concept_expired_status[concept]
                        bg_color = 'orange' if is_expired else 'lightgray'
                        text_color = 'black'
                        plt.annotate(concept, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), 
                                    fontsize=14, ha='center', va='center',
                                    bbox=dict(boxstyle="round,pad=0.6", fc=bg_color, ec="gray", alpha=0.95, linewidth=2))
                    
                    legend_elements = [
                        Patch(facecolor='lightgray', edgecolor='gray', label='Active Concepts'),
                        Patch(facecolor='orange', edgecolor='gray', label='Expired Concepts')
                    ]
                    plt.legend(handles=legend_elements, loc='upper right')
                    
                    plt.title("Concept Space Visualization (PCA fallback)", fontsize=18)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    sys.stderr.write(f"Fallback PCA visualization saved to: {save_path}\n")
                except Exception as e2:
                    sys.stderr.write(f"Error in fallback PCA visualization: {e2}\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "embedding_model": self.embedding_model,
            "concept_embeddings": {
                concept: embedding.to_dict() 
                for concept, embedding in self.concept_embeddings.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], client) -> 'ConceptSpaceModel':
        """Create from dictionary"""
        embedding_model = data.get("embedding_model", "openai")  # Default to openai for backward compatibility
        model = cls(client, embedding_model)
        model.concept_embeddings = {
            concept: ConceptEmbedding.from_dict(embedding_data)
            for concept, embedding_data in data["concept_embeddings"].items()
        }
        return model

class ConceptPool:
    def __init__(self, initial_concepts: List[str], client=None, embedding_model="openai", max_unsuccessful_uses=3):
        self.original_concepts = set(initial_concepts)  # Store original concepts separately
        self.concepts = set(initial_concepts)  # All active concepts including original ones
        self.all_concepts = set(initial_concepts)  # All concepts ever added (including expired)
        self.concept_history = []  # List of (generation, action, concepts) tuples
        self.action_history = []   # List of (generation, action) tuples
        self.concept_space_model = ConceptSpaceModel(client, embedding_model) if client else None
        self.max_unsuccessful_uses = max_unsuccessful_uses  # Maximum unsuccessful uses before expiry
        self.similarity_history = {}  # Dictionary to track similarity by generation
        
        # Initialize concept embeddings if model is available
        if self.concept_space_model:
            for concept in initial_concepts:
                self.concept_space_model.add_concept(concept)
            
            # Calculate and store initial similarity
            if len(initial_concepts) >= 2:
                self.similarity_history[0] = self.calculate_average_similarity(list(initial_concepts))
    
    def calculate_average_similarity(self, concepts: List[str]) -> float:
        """Calculate the average cosine similarity between all pairs of concepts"""
        if not self.concept_space_model or len(concepts) < 2:
            return 0.0
            
        # Calculate similarity for all pairs
        similarities = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                similarity = self.concept_space_model.get_similarity(concepts[i], concepts[j])
                similarities.append(similarity)
                
        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def update_similarity_history(self, generation: int) -> None:
        """Update the similarity history for the current generation"""
        # Use all concepts (active and expired)
        concepts = list(self.all_concepts)
        if len(concepts) >= 2 and self.concept_space_model:
            self.similarity_history[generation] = self.calculate_average_similarity(concepts)
    
    def visualize_similarity_evolution(self, save_path: str) -> None:
        """Visualize the evolution of average similarity over generations"""
        if not self.similarity_history:
            sys.stderr.write("No similarity history available for visualization\n")
            return
            
        generations = sorted(self.similarity_history.keys())
        similarities = [self.similarity_history[g] for g in generations]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Convert from 0-indexed to 1-indexed generations for display to match the numbering shown to users in messages.json
        display_generations = [g+1 for g in generations]
        
        # Plot similarity evolution on the first subplot
        ax1.plot(display_generations, similarities, marker='o', linestyle='-', linewidth=2, color='blue')
        ax1.set_xlabel('Generation', fontsize=14)
        ax1.set_ylabel('Average Cosine Similarity', fontsize=14)
        ax1.set_title('Evolution of Concept Similarity Over Generations', fontsize=16)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Highlight significant changes
        for i in range(1, len(generations)):
            change = similarities[i] - similarities[i-1]
            if abs(change) > 0.05:  # Threshold for significant change
                color = 'green' if change > 0 else 'red'
                ax1.plot([display_generations[i-1], display_generations[i]], [similarities[i-1], similarities[i]], 
                       linewidth=3, color=color, alpha=0.7)
        
        # Add annotations for significant events
        for i, gen in enumerate(generations):
            # Find actions at this generation
            actions = [action for g, action in self.action_history if g == gen]
            if actions and actions[0] != Action.ADD_NO_CONCEPT:
                # Mark generations where concepts were added
                ax1.annotate(actions[0].value, 
                           (display_generations[i], similarities[i]),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="orange", alpha=0.6))
                
                # Add a vertical line to highlight this generation
                ax1.axvline(x=display_generations[i], color='orange', linestyle='--', alpha=0.5)
        
        # Add a trend line
        if len(generations) > 1:
            z = np.polyfit(display_generations, similarities, 1)
            p = np.poly1d(z)
            ax1.plot(display_generations, p(display_generations), "r--", alpha=0.8, label=f"Trend: {z[0]:.4f}x + {z[1]:.4f}")
            ax1.legend()
        
        # Second subplot: Show concept count over generations
        concept_counts = []
        expired_counts = []
        
        # Track all concepts that have been added to the pool
        all_active_concepts = set()
        
        for gen in generations:
            # Get concepts active at this generation
            active_concepts = set()
            expired_in_this_gen = set()
            
            # If we have a concept space model, use it to check for expirations
            if self.concept_space_model:
                for concept, embedding in self.concept_space_model.concept_embeddings.items():
                    # Check if the concept was added by this generation
                    was_present = False
                    for g, action, concepts in self.concept_history:
                        if g > gen:
                            break
                        
                        if action in [Action.ADD_ONE_CONCEPT, Action.ADD_MULTIPLE_CONCEPTS]:
                            if concept in concepts:
                                was_present = True
                                all_active_concepts.add(concept)
                        
                        if action == Action.RECOMBINATION and concept in concepts:
                            was_present = True
                    
                    # If concept was present and expired in this generation, count it as expired
                    if was_present and embedding.is_expired and embedding.expired_at_generation == gen:
                        expired_in_this_gen.add(concept)
                    
                    # If concept was present and not expired yet (or expired in a future generation)
                    if was_present and (not embedding.is_expired or 
                                       (embedding.expired_at_generation is not None and 
                                        embedding.expired_at_generation > gen)):
                        active_concepts.add(concept)
            
            # Add original concepts to active concepts
            active_concepts.update(self.original_concepts)
            
            concept_counts.append(len(active_concepts))
            expired_counts.append(len(expired_in_this_gen))
        
        # Plot concept counts
        ax2.bar(display_generations, concept_counts, alpha=0.7, label='Active Concepts')
        ax2.bar(display_generations, expired_counts, bottom=concept_counts, alpha=0.7, color='red', label='Expired Concepts')
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Concept Count', fontsize=12)
        ax2.set_title('Concepts per Generation', fontsize=14)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        sys.stderr.write(f"Similarity evolution visualization saved to: {save_path}\n")
    
    def add_no_concept(self, generation: int) -> None:
        """Record that no new concept was added in this generation"""
        self.concept_history.append((generation, Action.ADD_NO_CONCEPT, []))
        self.action_history.append((generation, Action.ADD_NO_CONCEPT))
        # Update similarity history
        self.update_similarity_history(generation)
    
    def add_concept(self, concept: str, generation: int) -> None:
        """Add a new concept to the pool"""
        self.concepts.add(concept)
        self.all_concepts.add(concept)
        self.concept_history.append((generation, Action.ADD_ONE_CONCEPT, [concept]))
        self.action_history.append((generation, Action.ADD_ONE_CONCEPT))
        
        # Add to concept space model if available
        if self.concept_space_model:
            self.concept_space_model.add_concept(concept)
        
        # Update similarity history
        self.update_similarity_history(generation)
    
    def add_multiple_concepts(self, new_concepts: List[str], generation: int) -> None:
        """Add multiple new concepts to the pool"""
        for concept in new_concepts:
            self.concepts.add(concept)
            self.all_concepts.add(concept)
            
            # Add to concept space model if available
            if self.concept_space_model:
                self.concept_space_model.add_concept(concept)
        
        self.concept_history.append((generation, Action.ADD_MULTIPLE_CONCEPTS, new_concepts))
        self.action_history.append((generation, Action.ADD_MULTIPLE_CONCEPTS))
        
        # Update similarity history
        self.update_similarity_history(generation)
    
    def recombine_concepts(self, concepts: List[str], generation: int, improved_fitness: bool = False) -> None:
        """Record a recombination of existing concepts and update usage stats"""
        # Use the RECOMBINATION action enum instead of a string
        self.concept_history.append((generation, Action.RECOMBINATION, concepts))
        # We don't add this to action_history since it's not a user-selected action
        
        # Update usage counts in concept space model
        if self.concept_space_model:
            for concept in concepts:
                if concept in self.concept_space_model.concept_embeddings:
                    self.concept_space_model.concept_embeddings[concept].update_usage(
                        improved_fitness, 
                        concept in self.original_concepts
                    )
        
        # Check for concept expiry
        self.check_concept_expiry()
        
        # Update similarity history after potential concept expiry
        self.update_similarity_history(generation)
    
    def check_concept_expiry(self) -> List[str]:
        """Check for concepts that should expire and remove them from active pool"""
        expired_concepts = []
        
        if self.concept_space_model:
            # Get current generation from the most recent history entry
            current_generation = 0
            if self.concept_history:
                current_generation = self.concept_history[-1][0]  # (generation, action, concepts)
                
                # Note: current_generation is 0-indexed internally, but shown as 1-indexed to users
                # We store the 0-indexed version for internal consistency, but log the 1-indexed version
                display_generation = current_generation + 1
            
            for concept, embedding in self.concept_space_model.concept_embeddings.items():
                # Skip original concepts - they never expire
                if concept in self.original_concepts:
                    continue
                    
                # Check if concept should expire
                if (embedding.unsuccessful_uses >= self.max_unsuccessful_uses and 
                    not embedding.is_expired):
                    embedding.is_expired = True
                    embedding.expired_at_generation = current_generation + 1
                    if concept in self.concepts:
                        self.concepts.remove(concept)
                        expired_concepts.append(concept)
                        sys.stderr.write(f"Concept expired: '{concept}' after {embedding.unsuccessful_uses} unsuccessful uses in generation {display_generation}\n")
        
        return expired_concepts
    
    def get_concepts(self) -> List[str]:
        """Get all active concepts in the pool"""
        return list(self.concepts)
    
    def get_all_concepts(self) -> List[str]:
        """Get all concepts ever added to the pool (including expired)"""
        return list(self.all_concepts)
    
    def get_expired_concepts(self) -> List[str]:
        """Get list of expired concepts"""
        if not self.concept_space_model:
            return []
            
        expired = []
        for concept, embedding in self.concept_space_model.concept_embeddings.items():
            if embedding.is_expired:
                expired.append(concept)
        return expired
    
    def get_original_concepts(self) -> List[str]:
        """Get the original concepts"""
        return list(self.original_concepts)
    
    def is_original_concept(self, concept: str) -> bool:
        """Check if a concept is one of the original concepts"""
        return concept in self.original_concepts
    
    def is_expired_concept(self, concept: str) -> bool:
        """Check if a concept is expired"""
        if not self.concept_space_model:
            return False
            
        if concept not in self.concept_space_model.concept_embeddings:
            return False
            
        return self.concept_space_model.concept_embeddings[concept].is_expired
    
    def get_history(self) -> List[Tuple[int, Action, List[str]]]:
        """Get the concept history"""
        return self.concept_history
    
    def get_action_history(self) -> List[Tuple[int, Action]]:
        """Get the action history"""
        return self.action_history
    
    def get_last_action(self) -> Optional[Tuple[int, Action]]:
        """Get the last action taken"""
        return self.action_history[-1] if self.action_history else None
    
    def get_concept_clusters(self, n_clusters: int = 3) -> Dict[int, List[str]]:
        """Group concepts into semantic clusters"""
        if not self.concept_space_model:
            return {0: list(self.concepts)}
        
        return self.concept_space_model.get_concept_clusters(list(self.concepts), n_clusters)
    
    def get_concept_novelty(self, concept: str) -> float:
        """Get the novelty score of a concept compared to the pool"""
        if not self.concept_space_model:
            return 0.5  # Default value if no model
        
        existing = [c for c in self.concepts if c != concept]
        return self.concept_space_model.calculate_novelty(concept, existing)
    
    def suggest_new_concepts(self, n_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Suggest new concepts that would be novel additions to the pool"""
        if not self.concept_space_model:
            return []
        
        return self.concept_space_model.suggest_new_concept(list(self.concepts), n_suggestions)
    
    def visualize_concept_space(self, save_path: str, current_generation: int = None) -> None:
        """Visualize the concept space using t-SNE
        
        Args:
            save_path: Path to save the visualization
            current_generation: The generation number to use when determining expired concepts.
                               If None, uses the latest generation.
        """
        if not self.concept_space_model:
            sys.stderr.write("Concept space model not available for visualization\n")
            return
        
        # Get all concepts (active and expired)
        all_concepts = list(self.all_concepts)
        
        if len(all_concepts) < 2:
            sys.stderr.write("Need at least 2 concepts to visualize concept space\n")
            return
        
        # If current_generation is not provided, use the latest one
        if current_generation is None and self.concept_history:
            current_generation = self.concept_history[-1][0]
            
        # Create a dictionary mapping concepts to their expired status at the given generation
        concept_expired_status = {}
        for concept in all_concepts:
            if concept in self.concept_space_model.concept_embeddings:
                embedding = self.concept_space_model.concept_embeddings[concept]
                # A concept is expired at this generation if:
                # 1. It is marked as expired AND
                # 2. Its expired_at_generation is not None AND
                # 3. Its expired_at_generation is <= the current generation
                is_expired = (embedding.is_expired and 
                             embedding.expired_at_generation is not None and 
                             embedding.expired_at_generation <= current_generation)
                concept_expired_status[concept] = is_expired
            else:
                concept_expired_status[concept] = False
                
        # Pass the concepts and their expired status to the visualization method
        self.concept_space_model.visualize_concept_space(all_concepts, save_path, concept_expired_status)
    
    def generate_visualizations(self, base_path: str) -> None:
        """Generate all concept-related visualizations
        
        Args:
            base_path: Base path for saving visualizations (without file extension)
        """
        if not self.concept_space_model:
            sys.stderr.write("Concept space model not available for visualization\n")
            return
            
        # Get the latest generation number
        latest_generation = None
        if self.concept_history:
            latest_generation = self.concept_history[-1][0]
            
        # Generate concept space visualization (all concepts at latest generation)
        concept_space_path = f"{base_path}_concept_space.png"
        self.visualize_concept_space(concept_space_path, current_generation=latest_generation)
        
        # Generate similarity evolution visualization
        similarity_path = f"{base_path}_similarity_evolution.png"
        self.visualize_similarity_evolution(similarity_path)
        
        sys.stderr.write(f"All visualizations generated at {base_path}_*.png\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {
            "original_concepts": list(self.original_concepts),
            "concepts": list(self.concepts),
            "all_concepts": list(self.all_concepts),
            "concept_history": [(g, a.value, c) for g, a, c in self.concept_history],
            "action_history": [(g, a.value) for g, a in self.action_history],
            "max_unsuccessful_uses": self.max_unsuccessful_uses,
            "similarity_history": {str(k): float(v) for k, v in self.similarity_history.items()}
        }
        
        if self.concept_space_model:
            data["concept_space_model"] = self.concept_space_model.to_dict()
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], client=None, embedding_model="openai") -> 'ConceptPool':
        """Create from dictionary"""
        max_unsuccessful_uses = data.get("max_unsuccessful_uses", 3)
        pool = cls(data["original_concepts"], client, embedding_model, max_unsuccessful_uses)
        pool.concepts = set(data["concepts"])
        pool.all_concepts = set(data.get("all_concepts", data["concepts"]))
        
        # Reconstruct history
        pool.concept_history = [
            (g, Action(a), c) for g, a, c in data["concept_history"]
        ]
        pool.action_history = [
            (g, Action(a)) for g, a in data["action_history"]
        ]
        
        # Load similarity history if available
        if "similarity_history" in data:
            pool.similarity_history = {int(k): v for k, v in data["similarity_history"].items()}
        
        # Load concept space model if available
        if "concept_space_model" in data and client:
            # Get embedding model from saved data or use the provided one
            saved_embedding_model = data["concept_space_model"].get("embedding_model", embedding_model)
            pool.concept_space_model = ConceptSpaceModel.from_dict(data["concept_space_model"], client)
            pool.concept_space_model.embedding_model = saved_embedding_model
            
        return pool

class ConceptCombinationMemory:
    def __init__(self):
        self.combination_history = {}  # Maps concept combinations to their performance metrics
        self.concept_performance = {}  # Maps individual concepts to their performance metrics
        self.recency_weights = []  # Weights for recency bias
        
    def record_combination(self, concepts: List[str], fitness: float, aesthetic_score: float, 
                          originality_score: float, diversity_score: float, generation: int, 
                          improved_fitness: bool = False):
        """Record a concept combination and its performance metrics"""
        # Create a frozen set for the combination to use as a dictionary key
        combination_key = frozenset(concepts)
        
        # Record combination performance
        if combination_key not in self.combination_history:
            self.combination_history[combination_key] = {
                'occurrences': 0,
                'fitness_scores': [],
                'aesthetic_scores': [],
                'originality_scores': [],
                'diversity_scores': [],  # Add storage for diversity scores
                'generations': [],
                'improved_fitness_count': 0
            }
        
        history = self.combination_history[combination_key]
        history['occurrences'] += 1
        history['fitness_scores'].append(fitness)
        history['aesthetic_scores'].append(aesthetic_score)
        history['originality_scores'].append(originality_score)
        history['diversity_scores'].append(diversity_score)  # Always store diversity score now
        history['generations'].append(generation)
        if improved_fitness:
            history['improved_fitness_count'] = history.get('improved_fitness_count', 0) + 1
        
        # Update individual concept performance
        for concept in concepts:
            if concept not in self.concept_performance:
                self.concept_performance[concept] = {
                    'occurrences': 0,
                    'fitness_scores': [],
                    'aesthetic_scores': [],
                    'originality_scores': [],
                    'diversity_scores': [],  # Add storage for diversity scores
                    'generations': [],
                    'improved_fitness_count': 0
                }
            
            perf = self.concept_performance[concept]
            perf['occurrences'] += 1
            perf['fitness_scores'].append(fitness)
            perf['aesthetic_scores'].append(aesthetic_score)
            perf['originality_scores'].append(originality_score)
            perf['diversity_scores'].append(diversity_score)  # Always store diversity score
            perf['generations'].append(generation)
            if improved_fitness:
                perf['improved_fitness_count'] = perf.get('improved_fitness_count', 0) + 1
    
    def get_combination_performance(self, concepts: List[str]) -> Dict[str, Any]:
        """Get performance metrics for a specific combination"""
        combination_key = frozenset(concepts)
        return self.combination_history.get(combination_key, None)
    
    def get_concept_performance(self, concept: str) -> Dict[str, Any]:
        """Get performance metrics for a specific concept"""
        return self.concept_performance.get(concept, None)
    
    def get_best_combinations(self, n: int = 5) -> List[Tuple[Set[str], float]]:
        """Get the top N performing combinations based on average fitness"""
        if not self.combination_history:
            return []
        
        # Calculate average fitness for each combination
        avg_fitness = [(combo, sum(history['fitness_scores'])/len(history['fitness_scores']))
                      for combo, history in self.combination_history.items()]
        
        # Sort by average fitness (descending)
        sorted_combinations = sorted(avg_fitness, key=lambda x: x[1], reverse=True)
        
        return sorted_combinations[:n]
    
    def get_best_concepts(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the top N performing individual concepts based on average fitness"""
        if not self.concept_performance:
            return []
        
        # Calculate average fitness for each concept
        avg_fitness = [(concept, sum(perf['fitness_scores']) / len(perf['fitness_scores']))
                      for concept, perf in self.concept_performance.items()]
        
        # Sort by average fitness (descending)
        sorted_concepts = sorted(avg_fitness, key=lambda x: x[1], reverse=True)
        
        return sorted_concepts[:n]
    
    def predict_combination_fitness(self, concepts: List[str]) -> float:
        """Predict the fitness of a new combination based on historical performance"""
        # Check if this exact combination exists
        combination_key = frozenset(concepts)
        if combination_key in self.combination_history:
            history = self.combination_history[combination_key]
            return sum(history['fitness_scores']) / len(history['fitness_scores'])
        
        # If not, estimate based on individual concept performance
        concept_scores = []
        for concept in concepts:
            if concept in self.concept_performance:
                perf = self.concept_performance[concept]
                avg_score = sum(perf['fitness_scores']) / len(perf['fitness_scores'])
                concept_scores.append(avg_score)
            else:
                # For unknown concepts, use neutral score
                concept_scores.append(0.5)
        
        # Return average of concept scores if available
        if concept_scores:
            return sum(concept_scores) / len(concept_scores)
        return 0.5  # Default neutral prediction
    
    def get_concept_synergies(self) -> Dict[Tuple[str, str], float]:
        """Identify pairs of concepts that work well together"""
        if len(self.combination_history) < 2:
            return {}
        
        # Track all concept pairs and their joint performance
        pair_performance = {}
        
        for combination, history in self.combination_history.items():
            avg_fitness = sum(history['fitness_scores']) / len(history['fitness_scores'])
            
            # For each pair of concepts in this combination
            concepts = list(combination)
            for i in range(len(concepts)):
                for j in range(i+1, len(concepts)):
                    pair = (concepts[i], concepts[j]) if concepts[i] < concepts[j] else (concepts[j], concepts[i])
                    
                    if pair not in pair_performance:
                        pair_performance[pair] = {
                            'occurrences': 0,
                            'total_fitness': 0
                        }
                    
                    pair_performance[pair]['occurrences'] += 1
                    pair_performance[pair]['total_fitness'] += avg_fitness
        
        # Calculate average fitness for each pair
        synergies = {pair: perf['total_fitness'] / perf['occurrences'] 
                    for pair, perf in pair_performance.items() 
                    if perf['occurrences'] > 0}
        
        return synergies
    
    def suggest_promising_combinations(self, available_concepts: List[str], 
                                      n_suggestions: int = 5) -> List[Tuple[List[str], float]]:
        """Suggest promising combinations based on historical performance"""
        if not self.combination_history or not available_concepts:
            return []
        
        import itertools
        
        # Generate all possible combinations (up to a reasonable size)
        max_combination_size = min(5, len(available_concepts))
        potential_combinations = []
        
        for size in range(2, max_combination_size + 1):
            for combo in itertools.combinations(available_concepts, size):
                # Predict fitness for this combination
                predicted_fitness = self.predict_combination_fitness(combo)
                potential_combinations.append((list(combo), predicted_fitness))
        
        # Sort by predicted fitness
        sorted_combinations = sorted(potential_combinations, key=lambda x: x[1], reverse=True)
        
        return sorted_combinations[:n_suggestions]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        # Convert combination history, making sure all numeric values are Python native types
        combination_history_dict = {}
        for combo, history in self.combination_history.items():
            combo_key = ",".join(sorted(combo))
            history_copy = {}
            for key, value in history.items():
                if key in ['fitness_scores', 'aesthetic_scores', 'originality_scores', 'diversity_scores']:
                    # Convert possible NumPy values in lists to native Python floats
                    history_copy[key] = [float(score) for score in value]
                else:
                    history_copy[key] = value
            combination_history_dict[combo_key] = history_copy
            
        # Convert concept performance, making sure all numeric values are Python native types
        concept_performance_dict = {}
        for concept, perf in self.concept_performance.items():
            perf_copy = {}
            for key, value in perf.items():
                if key in ['fitness_scores', 'aesthetic_scores', 'originality_scores', 'diversity_scores']:
                    # Convert possible NumPy values in lists to native Python floats
                    perf_copy[key] = [float(score) for score in value]
                else:
                    perf_copy[key] = value
            concept_performance_dict[concept] = perf_copy
        
        return {
            "combination_history": combination_history_dict,
            "concept_performance": concept_performance_dict
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptCombinationMemory':
        """Create from dictionary"""
        memory = cls()
        
        # Restore combination history
        for combo_str, history in data["combination_history"].items():
            combo = frozenset(combo_str.split(","))
            memory.combination_history[combo] = history
        
        # Restore concept performance
        memory.concept_performance = data["concept_performance"]
        
        return memory