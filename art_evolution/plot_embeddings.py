import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from torchvision.models import ResNet152_Weights
from matplotlib import rcParams
import json
import glob
import argparse
from matplotlib.lines import Line2D

# Set up the plot style for a professional look
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 16

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, image_path
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None, image_path

def get_resnet_model():
    """Get ResNet model for embeddings"""
    sys.stderr.write("Loading ResNet model for embeddings...\n")
    
    # Load the ResNet152 model with the latest weights
    weights = ResNet152_Weights.DEFAULT
    model = models.resnet152(weights=weights)
    num_ftrs = model.fc.in_features
    
    # Modify the final fully connected layer
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 26)
    )
    
    # Load the saved model weights
    model_path = 'data/embedding/best_model_resnet_152.pth'
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Extract feature extractor (everything except the classification layer)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    
    # Define the transformation for ResNet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return feature_extractor, transform

def embed_images(image_paths, feature_extractor, transform, batch_size=32):
    """Generate embeddings for a list of image paths"""
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    embeddings = []
    valid_paths = []
    
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Processing images"):
            valid_batch_images = []
            valid_batch_paths = []
            
            for img, path in zip(images, paths):
                if img is not None:
                    valid_batch_images.append(img)
                    valid_batch_paths.append(path)
            
            if not valid_batch_images:
                continue
                
            valid_batch_images = torch.stack(valid_batch_images).to(device)
            outputs = feature_extractor(valid_batch_images)
            batch_embeddings = outputs.view(outputs.size(0), -1)
            
            embeddings.extend(batch_embeddings.cpu().numpy())
            valid_paths.extend(valid_batch_paths)
    
    return np.array(embeddings), valid_paths

def load_wikiart_embeddings(embeddings_path):
    """Load pre-computed WikiArt embeddings"""
    sys.stderr.write(f"Loading WikiArt embeddings from {embeddings_path}...\n")
    data = np.load(embeddings_path, allow_pickle=True).item()
    return data['embeddings'], data['labels']

def collect_generation_data(run_dir):
    """Collect data about all generations from a run directory"""
    # Find all artwork_info_*.json files in the info subdirectory
    info_dir = os.path.join(run_dir, "info")
    if not os.path.exists(info_dir):
        # Fall back to the old structure if info directory doesn't exist
        info_dir = run_dir
    
    info_files = sorted(glob.glob(os.path.join(info_dir, "artwork_info_*.json")), 
                        key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
    
    if not info_files:
        sys.stderr.write(f"No artwork info files found in {info_dir}\n")
        return [], [], [], []
    
    generations = []
    image_paths = []
    names = []
    fitness_scores = []
    
    for info_file in info_files:
        with open(info_file, 'r') as f:
            info = json.load(f)
            
        gen_num = info['generation']
        image_path = info['path']
        name = info['name']
        fitness = info['fitness']
        
        # Handle relative paths correctly
        if not os.path.isabs(image_path):
            # If the path is relative, make it absolute
            image_path = os.path.join(run_dir, image_path)
        
        # Check if the file exists
        if not os.path.exists(image_path):
            sys.stderr.write(f"Warning: Image file not found at {image_path}\n")
            # Try to find the image in the images subdirectory
            images_dir = os.path.join(run_dir, "images")
            if os.path.exists(images_dir):
                image_name = os.path.basename(image_path)
                alternative_path = os.path.join(images_dir, image_name)
                if os.path.exists(alternative_path):
                    sys.stderr.write(f"Found image at alternative path: {alternative_path}\n")
                    image_path = alternative_path
        
        generations.append(gen_num)
        image_paths.append(image_path)
        names.append(name)
        fitness_scores.append(fitness)
    
    return generations, image_paths, names, fitness_scores

def create_embedding_visualization(run_dir, wikiart_embeddings_path, output_dir=None):
    """Create visualization of embeddings for a run"""
    if output_dir is None:
        output_dir = run_dir
    
    # Get model and transform
    feature_extractor, transform = get_resnet_model()
    
    # Load WikiArt embeddings
    wikiart_embeddings, wikiart_labels = load_wikiart_embeddings(wikiart_embeddings_path)
    
    # Collect generation data
    generations, image_paths, names, fitness_scores = collect_generation_data(run_dir)
    
    # Generate embeddings for the generated images
    generated_embeddings, _ = embed_images(image_paths, feature_extractor, transform)
    
    # Combine all embeddings for dimensionality reduction
    all_embeddings = np.vstack([wikiart_embeddings, generated_embeddings])
    
    # Reduce dimensionality with PCA first (to 32 dimensions)
    sys.stderr.write("Performing PCA dimensionality reduction...\n")
    pca = PCA(n_components=32, random_state=42)
    all_embeddings_pca = pca.fit_transform(all_embeddings)
    
    # Split back into WikiArt and generated embeddings
    wikiart_embeddings_pca = all_embeddings_pca[:len(wikiart_embeddings)]
    generated_embeddings_pca = all_embeddings_pca[len(wikiart_embeddings):]
    
    # Further reduce to 2D with UMAP
    sys.stderr.write("Performing UMAP dimensionality reduction...\n")
    reducer = umap.UMAP(random_state=42)
    all_embeddings_2d = reducer.fit_transform(all_embeddings_pca)
    
    # Split back into WikiArt and generated embeddings
    wikiart_embeddings_2d = all_embeddings_2d[:len(wikiart_embeddings)]
    generated_embeddings_2d = all_embeddings_2d[len(wikiart_embeddings):]
    
    # Create DataFrames for plotting
    wikiart_df = pd.DataFrame({
        'UMAP1': wikiart_embeddings_2d[:, 0],
        'UMAP2': wikiart_embeddings_2d[:, 1],
        'Label': wikiart_labels,
        'Type': 'WikiArt'
    })
    
    generated_df = pd.DataFrame({
        'UMAP1': generated_embeddings_2d[:, 0],
        'UMAP2': generated_embeddings_2d[:, 1],
        'Generation': generations,
        'Name': names,
        'Fitness': fitness_scores,
        'Type': 'Generated'
    })
    
    # Create color palette for WikiArt styles
    unique_styles = sorted(wikiart_df['Label'].unique())
    style_colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set3
    style_color_map = {style: style_colors[i % len(style_colors)] for i, style in enumerate(unique_styles)}
    
    # Create a matplotlib-compatible color map (converting from Plotly colors if needed)
    mpl_style_color_map = {}
    for style, color in style_color_map.items():
        # Convert Plotly RGB string format to matplotlib format if needed
        if isinstance(color, str) and color.startswith('rgb'):
            # Extract RGB values from the string
            rgb_values = color.strip('rgb()').split(',')
            r, g, b = [float(v)/255 for v in rgb_values]
            mpl_style_color_map[style] = (r, g, b)
        else:
            mpl_style_color_map[style] = color
    
    # Create interactive plot with Plotly
    fig = go.Figure()
    
    # Add WikiArt points
    for style in unique_styles:
        style_df = wikiart_df[wikiart_df['Label'] == style]
        fig.add_trace(go.Scatter(
            x=style_df['UMAP1'],
            y=style_df['UMAP2'],
            mode='markers',
            marker=dict(
                size=5,
                color=style_color_map[style],
                opacity=0.6
            ),
            name=style,
            hoverinfo='text',
            text=style_df['Label'],
            legendgroup='WikiArt'
        ))
    
    # Add generated points with a color gradient based on generation
    fig.add_trace(go.Scatter(
        x=generated_df['UMAP1'],
        y=generated_df['UMAP2'],
        mode='markers',
        marker=dict(
            size=10,
            color=generated_df['Generation'],
            colorscale='Viridis',
            colorbar=dict(title='Generation'),
            line=dict(width=1, color='black')
        ),
        name='Generated Images',
        hoverinfo='text',
        text=generated_df.apply(lambda row: f"Gen {row['Generation']}: {row['Name']}<br>Fitness: {row['Fitness']:.4f}", axis=1),
        legendgroup='Generated'
    ))
    
    # Add trajectory line connecting generations in order
    fig.add_trace(go.Scatter(
        x=generated_df.sort_values('Generation')['UMAP1'],
        y=generated_df.sort_values('Generation')['UMAP2'],
        mode='lines',
        line=dict(
            color='black',
            width=2,
            dash='dot'
        ),
        name='Evolution Path',
        legendgroup='Generated'
    ))
    
    # Add arrows to show direction of evolution
    for i in range(len(generated_df) - 1):
        current = generated_df.iloc[i]
        next_gen = generated_df.iloc[i + 1]
        
        # Calculate the midpoint for the arrow
        mid_x = (current['UMAP1'] + next_gen['UMAP1']) / 2
        mid_y = (current['UMAP2'] + next_gen['UMAP2']) / 2
        
        # Calculate the direction vector
        dx = next_gen['UMAP1'] - current['UMAP1']
        dy = next_gen['UMAP2'] - current['UMAP2']
        
        # Normalize and scale
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx = dx / length
            dy = dy / length
        
        fig.add_annotation(
            x=mid_x,
            y=mid_y,
            ax=mid_x - dx * 0.5,
            ay=mid_y - dy * 0.5,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="black"
        )
    
    # Update layout
    fig.update_layout(
        title=f"Embedding Space Visualization - Evolution Trajectory",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend_title="Art Styles",
        width=1200,
        height=800,
        legend=dict(
            groupclick="toggleitem",
            tracegroupgap=5
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Save as HTML for interactive viewing
    html_path = os.path.join(output_dir, "embedding_visualization.html")
    fig.write_html(html_path)
    sys.stderr.write(f"Interactive visualization saved to {html_path}\n")
    
    # Create a static version for publications
    static_fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = static_fig.add_subplot(111)
    
    # Plot WikiArt points (with alpha for better visibility)
    for style in unique_styles:
        style_df = wikiart_df[wikiart_df['Label'] == style]
        ax.scatter(
            style_df['UMAP1'], 
            style_df['UMAP2'],
            s=10,
            alpha=0.3,
            label=style,
            color=mpl_style_color_map[style]  # Use matplotlib-compatible colors
        )
    
    # Plot generated points with size based on fitness
    scatter = ax.scatter(
        generated_df['UMAP1'],
        generated_df['UMAP2'],
        s=generated_df['Fitness'] * 100 + 50,  # Scale fitness for better visibility
        c=generated_df['Generation'],
        cmap='viridis',
        edgecolors='black',
        linewidths=1,
        zorder=10
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Generation')
    
    # Plot evolution path
    sorted_gen_df = generated_df.sort_values('Generation')
    ax.plot(
        sorted_gen_df['UMAP1'],
        sorted_gen_df['UMAP2'],
        'k--',
        linewidth=1.5,
        zorder=5
    )
    
    # Add generation labels
    for i, row in generated_df.iterrows():
        ax.annotate(
            f"{int(row['Generation'])}",
            (row['UMAP1'], row['UMAP2']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Add title and labels
    ax.set_title("Embedding Space Visualization - Evolution Trajectory")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    
    # Add legend but only for a subset of styles to avoid overcrowding
    legend_styles = unique_styles[:10]  # Limit to first 10 styles
    handles, labels = ax.get_legend_handles_labels()
    legend_indices = [labels.index(style) for style in legend_styles if style in labels]
    selected_handles = [handles[i] for i in legend_indices]
    selected_labels = [labels[i] for i in legend_indices]
    
    custom_line = Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                         markersize=10, label='Generated Images')
    selected_handles.append(custom_line)
    selected_labels.append('Generated Images')
    
    ax.legend(
        selected_handles, 
        selected_labels,
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.05),
        ncol=5,
        frameon=True,
        fontsize=8
    )
    
    # Save static figure
    static_path = os.path.join(output_dir, "embedding_visualization.png")
    plt.tight_layout()
    plt.savefig(static_path, dpi=300, bbox_inches='tight')
    plt.close()
    sys.stderr.write(f"Static visualization saved to {static_path}\n")
    
    return html_path, static_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embedding visualizations for AI Artist runs")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory containing the run data")
    parser.add_argument("--wikiart-embeddings", type=str, default="/u/alehe/Projects/AI-Scientist/data/embedding/embeddings.npy", 
                        help="Path to WikiArt embeddings file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save visualizations (defaults to run-dir)")
    
    args = parser.parse_args()
    
    create_embedding_visualization(args.run_dir, args.wikiart_embeddings, args.output_dir) 