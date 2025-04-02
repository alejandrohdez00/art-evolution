import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from concept_management import Action
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def plot_fitness_history(fitness_history, aesthetic_history, originality_history, diversity_history, save_dir):
    """
    Plot the fitness values and their components over generations and save the plot to the run directory.
    
    Args:
        fitness_history: List of combined fitness values for each generation
        aesthetic_history: List of aesthetic scores for each generation (0-1 scale)
        originality_history: List of originality scores for each generation (0-1 scale)
        diversity_history: List of diversity scores for each generation (0-1 scale)
        save_dir: Directory to save the plot
    """
    # Set seaborn style
    sns.set_theme(style="darkgrid")
    
    # Create a figure
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Create the generations list
    generations = list(range(1, len(fitness_history) + 1))
    
    # Normalize aesthetic scores to 0-1 range for plotting
    normalized_aesthetic_history = [score / 10.0 for score in aesthetic_history]
    
    # Plot all four metrics on the same plot
    plt.plot(generations, fitness_history, marker='o', linewidth=2.5, markersize=10, 
             color='#3366CC', label='Fitness')
    plt.plot(generations, normalized_aesthetic_history, marker='s', linewidth=2.5, markersize=8, 
             color='#33AA55', label='Aesthetic (0-1 scale)')
    plt.plot(generations, originality_history, marker='^', linewidth=2.5, markersize=8, 
             color='#AA3355', label='Originality (0-1 scale)')
    plt.plot(generations, diversity_history, marker='D', linewidth=2.5, markersize=8, 
             color='#FF9900', label='Diversity (0-1 scale)')
    
    # Add average lines
    avg_fitness = sum(fitness_history)/len(fitness_history)
    avg_normalized_aesthetic = sum(normalized_aesthetic_history)/len(normalized_aesthetic_history)
    avg_originality = sum(originality_history)/len(originality_history)
    avg_diversity = sum(diversity_history)/len(diversity_history)
    
    plt.axhline(y=avg_fitness, color='#3366CC', linestyle='--', linewidth=1.5, alpha=0.6,
                label=f'Avg Fitness: {avg_fitness:.2f}')
    plt.axhline(y=avg_normalized_aesthetic, color='#33AA55', linestyle='--', linewidth=1.5, alpha=0.6,
                label=f'Avg Aesthetic: {avg_normalized_aesthetic:.2f}')
    plt.axhline(y=avg_originality, color='#AA3355', linestyle='--', linewidth=1.5, alpha=0.6,
                label=f'Avg Originality: {avg_originality:.2f}')
    plt.axhline(y=avg_diversity, color='#FF9900', linestyle='--', linewidth=1.5, alpha=0.6,
                label=f'Avg Diversity: {avg_diversity:.2f}')
    
    # Customize plot
    plt.title('Fitness Scores Over Generations', fontsize=18, pad=20)
    plt.xlabel('Generation', fontsize=14, labelpad=10)
    plt.ylabel('Score', fontsize=14, labelpad=10)
    plt.xticks(generations)
    
    # Dynamically set y-axis limits based on data
    max_value = max(max(fitness_history), 1.0)  # At least 1.0 to show normalized scores properly
    plt.ylim(-0.05, max_value * 1.1)  # Add 10% padding above the maximum value
    
    # Add legend with better positioning and slightly different layout to fit all metrics
    plt.legend(fontsize=11, framealpha=0.9, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=4, fancybox=True, shadow=True)
    
    # Add grid but keep it subtle
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Tight layout
    plt.tight_layout(pad=3.0, rect=[0, 0.05, 1, 0.95])
    
    # Save the plot with high DPI for quality
    plot_path = os.path.join(save_dir, 'fitness_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    sys.stderr.write(f"Fitness history plot saved to: {plot_path}\n")
    return plot_path

def plot_phylogenetic_tree(concept_pool, fitness_history, save_dir):
    """
    Create a traditional phylogenetic tree visualization of concept evolution across generations
    with mandatory recombination in each generation.
    
    Args:
        concept_pool: ConceptPool object containing concept history
        fitness_history: List of fitness values for each generation
        save_dir: Directory to save the visualization
    """
    # Get concept history from pool
    concept_history = concept_pool.get_history()
    if not concept_history:
        sys.stderr.write("No concept history available to plot phylogenetic tree\n")
        return None
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Track concept nodes by generation
    concepts_by_gen = {}
    
    # Track recombination nodes by generation
    recombination_by_gen = {}
    
    # Get and add original concepts
    original_concepts = concept_pool.get_original_concepts()
    concepts_by_gen[0] = original_concepts
    
    for concept in original_concepts:
        G.add_node(concept, type='concept', generation=0, is_original=True)
    
    # Process history to build the tree
    current_gen = 0
    action_entries = []
    recombination_entries = []
    
    # First, separate action entries from recombination entries and group by generation
    for entry in concept_history:
        gen, action_type, concepts = entry
        
        # Group entries by generation
        if gen > current_gen:
            # Process previous generation's entries
            if action_entries and recombination_entries:
                process_generation_entries(G, concepts_by_gen, recombination_by_gen, 
                                          current_gen, action_entries, recombination_entries)
            
            # Reset for new generation
            current_gen = gen
            action_entries = []
            recombination_entries = []
        
        # Separate action entries from recombination entries
        if action_type == Action.RECOMBINATION:
            recombination_entries.append((gen, action_type, concepts))
        else:
            action_entries.append((gen, action_type, concepts))
    
    # Process the last generation
    if action_entries and recombination_entries:
        process_generation_entries(G, concepts_by_gen, recombination_by_gen, 
                                  current_gen, action_entries, recombination_entries)
    
    # Calculate node positions for a traditional tree layout
    pos = create_traditional_tree_layout(G, concepts_by_gen, recombination_by_gen)
    
    # Create the plot with a larger figure size
    plt.figure(figsize=(28, 20), dpi=300)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.7, 
                          edge_color='black', arrows=True,
                          arrowstyle='->', arrowsize=20)
    
    # Draw original concept nodes - make them larger
    original_nodes = [n for n, d in G.nodes(data=True) if d.get('is_original', False)]
    if original_nodes:
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=original_nodes,
                              node_color='#ff9900',  # Orange for original concepts
                              node_size=1500,  # Increased size
                              node_shape='s',  # Square for original
                              alpha=0.9)
    
    # Draw new concept nodes with color gradient based on generation
    new_nodes = [n for n, d in G.nodes(data=True) 
                if d.get('type') == 'concept' and not d.get('is_original', False)]
    
    if new_nodes:
        # Get the maximum generation number for color scaling
        max_gen = max([d.get('generation', 0) for n, d in G.nodes(data=True) 
                      if d.get('type') == 'concept' and not d.get('is_original', False)], default=1)
        
        # Create a better color gradient from blue to red
        cmap = plt.cm.viridis  # Using viridis colormap (blue to yellow-green)
        node_colors = []
        
        for node in new_nodes:
            gen = G.nodes[node].get('generation', 0)
            color_val = gen / max_gen if max_gen > 0 else 0.5
            node_colors.append(cmap(color_val))
        
        # Draw with larger node size
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=new_nodes,
                              node_color=node_colors,
                              node_size=1200,  # Increased size
                              node_shape='o',  # Circle for new concepts
                              alpha=0.9)
    
    # Draw recombination nodes
    recombination_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'recombination']
    if recombination_nodes:
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=recombination_nodes,
                              node_color='#ff5555',  # Red for recombinations
                              node_size=800,  # Increased size
                              node_shape='d',  # Diamond for recombinations
                              alpha=0.8)
    
    # Add labels directly on the nodes for better visibility
    # Create separate label dictionaries for original and new concepts
    original_labels = {node: node for node in original_nodes}
    new_labels = {node: node for node in new_nodes}
    
    # Draw labels for original concepts
    if original_labels:
        try:
            nx.draw_networkx_labels(G, pos, labels=original_labels,
                                   font_size=14, font_weight='bold',
                                   font_family='sans-serif',
                                   font_color='black')
        except KeyError as e:
            sys.stderr.write(f"Warning: KeyError when drawing original labels: {e}\n")
    
    # Draw labels for new concepts
    if new_labels:
        try:
            nx.draw_networkx_labels(G, pos, labels=new_labels,
                                   font_size=14, font_weight='bold',
                                   font_family='sans-serif',
                                   font_color='black')
        except KeyError as e:
            sys.stderr.write(f"Warning: KeyError when drawing new concept labels: {e}\n")
    
    # Create a custom colorbar to show the generation gradient
    if new_nodes:
        # Create a separate axis for the colorbar
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)
        
        # Create a colorbar with generation numbers
        norm = plt.Normalize(1, max_gen)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('Generation', fontsize=16)
    
    # Add legend with larger markers
    plt.plot([0], [0], 's', color='#ff9900', markersize=20, label='Original Concepts')
    plt.plot([0], [0], 'o', color=cmap(0.2), markersize=20, label='Early Generation Concepts')
    plt.plot([0], [0], 'o', color=cmap(0.8), markersize=20, label='Later Generation Concepts')
    plt.plot([0], [0], 'd', color='#ff5555', markersize=20, label='Recombinations')
    
    # Customize plot
    plt.title('Concept Evolution with Mandatory Recombination', fontsize=28)
    plt.legend(fontsize=18, loc='upper left', bbox_to_anchor=(0.01, 0.99))
    plt.axis('off')
    plt.tight_layout()
    
    # Save and close
    tree_path = os.path.join(save_dir, 'concept_phylogeny.png')
    plt.savefig(tree_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    sys.stderr.write(f"Phylogenetic tree visualization saved to: {tree_path}\n")
    return tree_path

def process_generation_entries(G, concepts_by_gen, recombination_by_gen, generation, 
                              action_entries, recombination_entries):
    """
    Process a generation's action and recombination entries to build the graph.
    
    Args:
        G: NetworkX DiGraph object
        concepts_by_gen: Dictionary mapping generation numbers to lists of concepts
        recombination_by_gen: Dictionary mapping generation numbers to lists of recombination nodes
        generation: Current generation number
        action_entries: List of action entries for this generation
        recombination_entries: List of recombination entries for this generation
    """
    current_gen = generation + 1  # Adjust generation number for display
    
    if current_gen not in concepts_by_gen:
        concepts_by_gen[current_gen] = []
    
    if current_gen not in recombination_by_gen:
        recombination_by_gen[current_gen] = []
    
    # Process action entries first
    for gen, action_type, concepts in action_entries:
        if action_type == Action.ADD_ONE_CONCEPT:
            # Add new concept node
            new_concept = concepts[0]
            if new_concept not in G:
                G.add_node(new_concept, type='concept', generation=current_gen, is_original=False)
                concepts_by_gen[current_gen].append(new_concept)
                
                # Connect to a concept from previous generation
                if current_gen > 1 and len(concepts_by_gen.get(current_gen-1, [])) > 0:
                    parent_concept = concepts_by_gen[current_gen-1][0]
                    G.add_edge(parent_concept, new_concept)
        
        elif action_type == Action.ADD_MULTIPLE_CONCEPTS:
            # Add multiple new concept nodes
            for new_concept in concepts:
                if new_concept not in G:
                    G.add_node(new_concept, type='concept', generation=current_gen, is_original=False)
                    concepts_by_gen[current_gen].append(new_concept)
                    
                    # Connect to a concept from previous generation
                    if current_gen > 1 and len(concepts_by_gen.get(current_gen-1, [])) > 0:
                        parent_concept = concepts_by_gen[current_gen-1][0]
                        G.add_edge(parent_concept, new_concept)
    
    # Then process recombination entries
    for idx, (gen, action_type, concepts) in enumerate(recombination_entries):
        # Create a recombination node with unique identifier
        recomb_node = f"recomb_gen{current_gen}_{idx}"
        G.add_node(recomb_node, type='recombination', generation=current_gen)
        recombination_by_gen[current_gen].append(recomb_node)
        
        # Connect all used concepts to the recombination node
        for concept in concepts:
            if concept in G:
                G.add_edge(concept, recomb_node)

def create_traditional_tree_layout(G, concepts_by_gen, recombination_by_gen):
    """
    Custom function to create a traditional hierarchical tree layout with recombination nodes.
    """
    pos = {}
    
    # Get all generations from both dictionaries
    all_gens = set(concepts_by_gen.keys()).union(set(recombination_by_gen.keys()))
    if not all_gens:
        sys.stderr.write("Warning: No generations found in layout data\n")
        return {node: (0, 0) for node in G.nodes()}  # Default positions
        
    max_gen = max(all_gens)
    
    # Calculate total width and height for the layout
    width = 20  # Horizontal spacing between generations
    
    # First, position concept nodes by generation (x-coordinate)
    for gen, concepts_list in concepts_by_gen.items():
        x_pos = gen * width  # Horizontal position based on generation
        
        # Skip empty concept lists
        if not concepts_list:
            continue
            
        # Sort concepts alphabetically for consistent ordering
        concepts_list.sort()
        
        # Calculate y-positions for each concept in this generation
        total_concepts = len(concepts_list)
        if total_concepts > 1:
            height_per_concept = 20  # Vertical spacing between concepts
            start_y = -(total_concepts - 1) * height_per_concept / 2
            
            for i, concept in enumerate(concepts_list):
                y_pos = start_y + i * height_per_concept
                pos[concept] = (x_pos, y_pos)
        elif total_concepts == 1:
            # If only one concept in this generation, place it at the center
            pos[concepts_list[0]] = (x_pos, 0)
    
    # Then, position recombination nodes
    for gen, recomb_list in recombination_by_gen.items():
        x_pos = gen * width + width/2  # Position recombination nodes between generations
        
        # Skip empty recombination lists
        if not recomb_list:
            continue
        
        # For each recombination node
        for i, recomb in enumerate(recomb_list):
            # Place at the center vertically
            pos[recomb] = (x_pos, 0)
    
    # Ensure all nodes have positions
    for node in G.nodes():
        if node not in pos:
            # If a node doesn't have a position, place it at the origin
            # This is a fallback to prevent errors
            pos[node] = (0, 0)
            sys.stderr.write(f"Warning: Node '{node}' had no position assigned, using default (0,0)\n")
    
    return pos

def plot_concept_artwork_graph(concept_history, artwork_info_list, save_dir):
    """
    Create a bipartite graph visualization showing the relationships between concepts and artworks,
    where artwork nodes are represented by the actual artwork images.
    
    Args:
        concept_history: List of (generation, action_type, concepts) tuples from the concept pool
        artwork_info_list: List of artwork info dictionaries containing name, generation, concepts_used, path, etc.
        save_dir: Directory to save the visualization
    """
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Track all concepts seen
    all_concepts = set()
    
    # Add artwork nodes and connect to concepts
    for artwork in artwork_info_list:
        artwork_name = artwork["name"]
        generation = artwork["generation"]
        concepts_used = artwork["concepts_used"]
        artwork_path = artwork["path"]  # Path to the artwork image
        
        # If the path is relative, make it absolute
        if not os.path.isabs(artwork_path):
            artwork_path = os.path.join(save_dir, "..", artwork_path)
        
        # Add artwork node with path information
        G.add_node(artwork_name, type='artwork', generation=generation, path=artwork_path)
        
        # Add concept nodes and edges from concepts to artwork
        for concept in concepts_used:
            all_concepts.add(concept)
            if concept not in G:
                G.add_node(concept, type='concept')
            G.add_edge(concept, artwork_name)
    
    # Create a figure with increased height to allow more space for the larger images
    # and to prevent tight layout warnings
    plt.figure(figsize=(16, 14), dpi=300)
    
    # Ensure adequate margins to avoid tight layout warnings
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Use a spring layout with more iteration and higher repulsion to spread out nodes
    pos = nx.spring_layout(G, k=0.7, iterations=100, seed=42)
    
    # Draw edges first so they're underneath the nodes
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray', 
                          arrows=True, arrowstyle='->', arrowsize=10)
    
    # Draw concept nodes with increased size
    concept_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'concept']
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=concept_nodes,
                          node_color='#3366CC',  # Blue for concepts
                          node_size=1000,  # Adjusted size
                          alpha=0.8,
                          node_shape='o')  # Circle for concepts
    
    # Add concept labels with increased font size
    nx.draw_networkx_labels(G, pos, 
                           labels={n: n for n in concept_nodes},
                           font_size=13,  # Adjusted font size
                           font_weight='bold',
                           font_family='sans-serif')
    
    # Get artwork nodes
    artwork_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'artwork']
    
    # Get the axes object for adding images
    ax = plt.gca()
    
    # Calculate a much larger zoom factor for images
    node_size = 5000
    zoom_factor = 0.6  # Significantly increased from 0.3
    image_size = np.sqrt(node_size) * zoom_factor
    
    # Find max generation for color normalization
    if artwork_nodes:
        max_gen = max([G.nodes[n]['generation'] for n in artwork_nodes])
    else:
        max_gen = 1
    
    # Draw artwork nodes with their images
    for node in artwork_nodes:
        # Position of the node
        x, y = pos[node]
        
        # Get artwork path and generation
        image_path = G.nodes[node]['path']
        generation = G.nodes[node]['generation']
        
        try:
            # Load the image
            img = plt.imread(image_path)
            if img.shape[2] == 4:  # If RGBA, convert to RGB
                img = img[:, :, :3]
            
            # Create OffsetImage with greatly increased zoom
            imagebox = OffsetImage(img, zoom=image_size/img.shape[0])
            imagebox.image.axes = ax
            
            # Create annotation box with thicker border
            ab = AnnotationBbox(
                imagebox, (x, y),
                xycoords='data',
                pad=0.0,
                frameon=True,
                bboxprops=dict(
                    edgecolor=plt.cm.viridis(generation / max_gen),
                    linewidth=6  # Thicker border
                )
            )
            
            # Add the AnnotationBbox to the axes
            ax.add_artist(ab)
            
            # Add a clearer generation label directly on the artwork border
            plt.text(x, y - image_size/1.7, f"Gen {generation+1}", 
                    horizontalalignment='center',
                    verticalalignment='top',
                    fontsize=12,  # Larger font
                    fontweight='bold',
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round', pad=0.4))
            
        except Exception as e:
            sys.stderr.write(f"Error loading image for {node}: {e}\n")
            # Fallback: draw a placeholder node with larger size
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=[node],
                                  node_color=[plt.cm.viridis(generation / max_gen)],
                                  node_size=node_size*1.5,  # Increased size
                                  alpha=0.9,
                                  node_shape='s')
            
            # Add the artwork name as a label
            nx.draw_networkx_labels(G, pos, 
                                  labels={node: node},
                                  font_size=12, 
                                  font_weight='bold',
                                  font_family='sans-serif')
    
    # Create a more visible legend for the color gradient
    if artwork_nodes:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, max_gen))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Generation Number', fontsize=14, fontweight='bold')
    
    # Add legend items with larger markers
    plt.plot([], [], 'o', color='#3366CC', markersize=15, label='Concept')
    plt.plot([], [], 's', color='gray', markersize=15, label='Artwork (Image)')
    
    # Add title with larger font
    plt.title('Concept-Artwork Relationship Graph', fontsize=24, pad=20, fontweight='bold')
    plt.legend(fontsize=14, loc='upper right', framealpha=0.9)
    
    plt.axis('off')
    
    # Add a white background to make the graph more visible
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    
    # Save the visualization with a tight bounding box
    graph_path = os.path.join(save_dir, 'concept_artwork_graph.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    sys.stderr.write(f"Concept-artwork relationship graph saved to: {graph_path}\n")
    return graph_path