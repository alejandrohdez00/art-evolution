import argparse
import json
import os
import time
from datetime import datetime
import sys
import openai
from typing import List, Dict, Optional, Tuple, Any, Set
import numpy as np
import matplotlib.pyplot as plt

from concept_management import ConceptPool, ConceptCombinationMemory, Action
from utils import generate_artwork, validate_prompt, get_fitness_function
from visualization import plot_fitness_history, plot_phylogenetic_tree, plot_concept_artwork_graph

# Constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
GPT_MODEL = "gpt-4o-2024-08-06"


SYSTEM_PROMPT = """You are an AI artistic innovator tasked with conceptualizing unprecedented paintings by creating novel combinations of ideas. Starting with an initial list of concepts, explore combinations no human artist could imagine due to your freedom from temporal, cultural, and geographic limitations. Always preserve and incorporate all the original concepts provided - none can be removed or erased.

IMPORTANT: The original concept list MUST ALWAYS be included in your artwork, regardless of which action you choose. These concepts are mandatory and cannot be omitted.

In each generation, you must:
1. FIRST choose ONE of three possible actions:
   - ADD_NO_CONCEPT: Don't add any new concept, just recombine existing ones
   - ADD_ONE_CONCEPT: Introduce a completely new concept to the pool
   - ADD_MULTIPLE_CONCEPTS: Introduce multiple new concepts to the pool at once

2. THEN perform MANDATORY RECOMBINATION:
   - Create a novel combination using existing concepts from the pool
   - MUST include ALL original concepts in the recombination
   - Can include additional concepts from the pool up to 4

CONCEPT EXPIRY RULES:
- Concepts that are used in 3 consecutive generations without improving fitness will EXPIRE and be removed from the pool
- Expired concepts cannot be used or reintroduced
- Original concepts never expire and must always be included

FITNESS COMPONENTS:
- Aesthetic Score (0-1): Measures the visual appeal of the artwork using an aesthetic prediction model. Higher is better.
- Originality Score (0-1): Measures how different the generated artwork is from existing art in the WikiArt dataset. Higher values indicate more original works.
- Diversity Score (0-1): Measures the semantic diversity between concepts used. 0 means all concepts are semantically similar, while 1 means they are maximally different. Higher diversity scores are typically better.
- Overall Fitness: A combined score of the above components. Your goal is to maximize this value.

When you respond, output a JSON with the following structure:
{
    "thought": "Your thought process for choosing the action and designing the artwork",
    "action": "add_no_concept" or "add_one_concept" or "add_multiple_concepts",
    "name": "Name of your artwork",
    "concepts_used": ["List of concepts used in the RECOMBINATION - MUST include all original concepts"],
    "new_concept": "The new concept (only if action is add_one_concept)",
    "new_concepts": ["List of new concepts (only if action is add_multiple_concepts, must be non-empty)"],
    "prompt": "The exact prompt to generate the artwork, When describing scenes, use exact colors and clear spatial relationships of the elements. Specify physical characteristics rather than subjective qualities. Define lighting sources and conditions precisely. Include environmental details with technical accuracy. Replace atmospheric/emotional language with observable features. Focus on what the elements ARE rather than how they make you feel."
}

IMPORTANT: Do not include duplicate keys in your JSON response. Each key should appear exactly once.

For ADD_NO_CONCEPT:
- Focus on recombining existing concepts in a novel way
- Explain why no new concept is needed for this generation

For ADD_ONE_CONCEPT:
- Choose a concept that has never been combined with the existing pool
- Explain why this new concept creates an interesting combination
- The new concept must be different from any existing ones

For ADD_MULTIPLE_CONCEPTS:
- Choose multiple new concepts that work well with the existing pool
- Explain why these new concepts create interesting combinations
- The new concepts must be different from any existing ones

HISTORICAL PERFORMANCE:
- When provided with historical performance data, use it to guide your decisions
- Consider using concepts or combinations that have performed well in the past
- Balance between exploiting known successful concepts and exploring new possibilities
- If a concept has consistently low performance, consider avoiding it unless you have a compelling reason to use it
- Look for synergies between concepts that have worked well together

The user will provide:
1. The current concept pool
2. The generation number
3. The previous fitness scores (overall and component-wise)
4. The previous action taken
5. Concept clusters and their semantic relationships
6. Historical performance data (when available)
7. List of expired concepts (if any)

Your goal is to maximize fitness by choosing the most promising action and creating compelling artwork while always incorporating all original concepts.
"""

def get_first_prompt(concepts: List[str]):
    """Return the first prompt to start the evolution"""
    return f"""
These are the concepts that start the evolution:
{concepts}

Please generate the next iteration. You can add or keep the same concepts or modify or change the generated ones. You cannot remove the original concepts. Pay attention to how the fitness is affected by the concepts you add or modify.
"""

def run_evolution(args):
    """Run the art evolution process"""
    # Replace colons with underscores or another valid character, needed in Windows OS
    now = str(datetime.now()).replace(" ", "_").replace(":", "-")
    
    # Format the concept list for the folder name
    concepts_str = "-".join(args.concepts)
    
    if not args.no_logging:
        save_dir = f"runs/{concepts_str}-{now}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Create subdirectories for the run
        images_dir = os.path.join(save_dir, "images")
        info_dir = os.path.join(save_dir, "info")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(info_dir, exist_ok=True)

    config = {
        "NUM_GENERATIONS": args.num_generations,
        "SAVE_DIR": save_dir if not args.no_logging else None,
        "IMAGES_DIR": images_dir if not args.no_logging else None,
        "INFO_DIR": info_dir if not args.no_logging else None,
        "NOW": now,
        "RESUME": args.resume,
        "CONCEPTS": args.concepts,
        "EMBEDDING_MODEL": args.embedding_model,
        "NO_HISTORICAL_INFORMATION": args.no_historical_information,
        "FITNESS_FUNCTION": args.fitness_function,
    }
    
    # Get the selected fitness function
    evaluate_fitness = get_fitness_function(args.fitness_function)
    
    # Initialize OpenAI client
    client = openai.OpenAI()
    
    # Initialize messages for the LLM
    if not args.resume:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_first_prompt(args.concepts)},
        ]
    else:
        messages = json.load(open(args.resume, "r"))
        assert messages[-1]["role"] == "user", "Last message must be user"

    # Load archive if using baselines
    if args.do_baselines:
        if args.resume:
            archive_path = os.path.dirname(args.resume) + "/archive.json"
        else:
            archive_path = "archive.json"  # Default path
        
        try:
            with open(archive_path, "r") as f:
                archive = json.load(f)
            if len(archive) < args.num_generations:
                sys.stderr.write(f"Warning: Archive contains only {len(archive)} entries, but {args.num_generations} generations requested\n")
        except FileNotFoundError:
            sys.stderr.write(f"Error: Archive file not found at {archive_path}\n")
            sys.stderr.write("Cannot run with --do-baselines without an archive file\n")
            exit(1)

    # Print configuration information
    sys.stderr.write(f"\n=== AI Artist Evolution ===\n")
    sys.stderr.write(f"Starting with concepts: {args.concepts}\n")
    sys.stderr.write(f"Number of generations: {args.num_generations}\n")
    sys.stderr.write(f"Embedding model: {args.embedding_model}\n")
    sys.stderr.write(f"Fitness function: {args.fitness_function}\n")
    if args.resume:
        sys.stderr.write(f"Resuming from: {args.resume}\n")
    sys.stderr.write(f"Saving results to: {save_dir if not args.no_logging else 'Disabled'}\n")
    sys.stderr.write("===========================\n")

    # Initialize concept pool with OpenAI client for embeddings
    concept_pool = ConceptPool(args.concepts, client, args.embedding_model, args.max_unsuccessful_uses)
    
    # Initialize concept memory for tracking historical performance
    concept_memory = ConceptCombinationMemory()
    
    # Try to load existing memory if resuming
    if args.resume:
        memory_path = os.path.dirname(args.resume) + "/concept_memory.json"
        try:
            with open(memory_path, "r") as f:
                memory_data = json.load(f)
                concept_memory = ConceptCombinationMemory.from_dict(memory_data)
            sys.stderr.write(f"Loaded concept memory from: {memory_path}\n")
            
            # Also load concept pool if available
            pool_path = os.path.dirname(args.resume) + "/concept_pool.json"
            if os.path.exists(pool_path):
                with open(pool_path, "r") as f:
                    pool_data = json.load(f)
                    concept_pool = ConceptPool.from_dict(pool_data, client, args.embedding_model)
                sys.stderr.write(f"Loaded concept pool from: {pool_path}\n")
        except FileNotFoundError:
            sys.stderr.write(f"No existing concept memory found at {memory_path}, starting fresh\n")
    
    # Make sure we have the initial similarity recording for generation 0
    if 0 not in concept_pool.similarity_history and len(args.concepts) >= 2:
        concept_pool.update_similarity_history(0)
        sys.stderr.write(f"Initialized similarity tracking for generation 0\n")
    
    # Initialize fitness for first iteration
    fitness = 0
    fitness_history = []
    aesthetic_history = []
    originality_history = []
    diversity_history = []
    
    for i in range(args.num_generations):
        sys.stderr.write(f"\n--- Generation {i+1}/{args.num_generations} ---\n")
        t_start = time.time()
        
        # Get current state for the LLM
        current_concepts = concept_pool.get_concepts()
        original_concepts = concept_pool.get_original_concepts()
        last_action = concept_pool.get_last_action()
        last_action_str = f"Last action: {last_action[1].value}" if last_action else "No previous action"
        
        # Get concept space information
        concept_clusters = concept_pool.get_concept_clusters(n_clusters=min(3, len(current_concepts)))
        cluster_info = "\n".join([f"Cluster {k}: {v}" for k, v in concept_clusters.items()])
        
        # GENERATE PROMPT
        if not args.do_baselines:
            for _ in range(API_MAX_RETRY):
                try:
                    # Get historical performance data
                    best_combinations = concept_memory.get_best_combinations(3)
                    best_concepts = concept_memory.get_best_concepts(5)
                    
                    # Format historical data for LLM (only if not in no-historical-information mode)
                    best_combinations_str = ""
                    best_concepts_str = ""
                    historical_performance = ""
                    
                    if not config["NO_HISTORICAL_INFORMATION"]:
                        if best_combinations:
                            combination_details = []
                            for combo, score in best_combinations:
                                # Get detailed performance metrics for this combination if available
                                combo_performance = concept_memory.get_combination_performance(list(combo))
                                if combo_performance:
                                    avg_aesthetic = sum(combo_performance['aesthetic_scores']) / len(combo_performance['aesthetic_scores'])
                                    avg_originality = sum(combo_performance['originality_scores']) / len(combo_performance['originality_scores'])
                                    
                                    # Diversity scores may not be available for older runs
                                    avg_diversity = 0.0
                                    if 'diversity_scores' in combo_performance and combo_performance['diversity_scores']:
                                        avg_diversity = sum(combo_performance['diversity_scores']) / len(combo_performance['diversity_scores'])
                                    
                                    combo_detail = f"Combination: {list(combo)}, Avg Fitness: {score:.2f} (Aesthetic: {avg_aesthetic:.2f}, Originality: {avg_originality:.2f}, Diversity: {avg_diversity:.2f})"
                                else:
                                    combo_detail = f"Combination: {list(combo)}, Avg Fitness: {score:.2f}"
                                
                                combination_details.append(combo_detail)
                            
                            best_combinations_str = "\n".join(combination_details)
                        
                        if best_concepts:
                            concept_details = []
                            for concept, score in best_concepts:
                                # Get detailed performance metrics for this concept if available
                                concept_performance = concept_memory.get_concept_performance(concept)
                                if concept_performance:
                                    avg_aesthetic = sum(concept_performance['aesthetic_scores']) / len(concept_performance['aesthetic_scores'])
                                    avg_originality = sum(concept_performance['originality_scores']) / len(concept_performance['originality_scores'])
                                    
                                    # Diversity scores may not be available for older runs
                                    avg_diversity = 0.0
                                    if 'diversity_scores' in concept_performance and concept_performance['diversity_scores']:
                                        avg_diversity = sum(concept_performance['diversity_scores']) / len(concept_performance['diversity_scores'])
                                    
                                    concept_detail = f"Concept: {concept}, Avg Fitness: {score:.2f} (Aesthetic: {avg_aesthetic:.2f}, Originality: {avg_originality:.2f}, Diversity: {avg_diversity:.2f})"
                                else:
                                    concept_detail = f"Concept: {concept}, Avg Fitness: {score:.2f}"
                                
                                concept_details.append(concept_detail)
                            
                            best_concepts_str = "\n".join(concept_details)
                        
                        # Prepare historical performance section
                        if best_combinations or best_concepts:
                            historical_performance = f"""
Historical performance:
Top performing combinations:
{best_combinations_str}

Top performing concepts:
{best_concepts_str}
"""
                    
                    # Get expired concepts
                    expired_concepts = concept_pool.get_expired_concepts()
                    expired_concepts_str = ""
                    if expired_concepts:
                        expired_concepts_str = f"\nExpired concepts (cannot be used): {expired_concepts}"
                    
                    # Calculate normalized aesthetic score for display
                    normalized_aesthetic = 0.0
                    if i > 0 and aesthetic_history:
                        normalized_aesthetic = aesthetic_history[-1] / 10.0
                    
                    # Create the user message for this generation
                    if i == 0:
                        # First generation has no previous fitness scores
                        fitness_info = "No previous fitness scores (first generation)"
                    else:
                        # Format fitness information with all components
                        fitness_info = f"""Previous fitness scores:
- Overall Fitness: {fitness:.2f}
- Aesthetic Score: {normalized_aesthetic:.2f}
- Originality Score: {originality_history[-1]:.2f}
- Diversity Score: {diversity_history[-1]:.2f}"""
                    
                    user_message = f"""
Current concept pool: {current_concepts}
Original concepts (MUST be included): {original_concepts}
Generation: {i+1}
{fitness_info}
{last_action_str}{expired_concepts_str}

Concept clusters:
{cluster_info}
{historical_performance}
Please generate the next artwork idea."""

                    # Add the user message to the messages array
                    messages.append({"role": "user", "content": user_message})
                    
                    # Get the completion from the LLM
                    completion = client.chat.completions.create(
                        model=GPT_MODEL,
                        messages=messages,
                        max_tokens=2048,
                        n=1,
                        temperature=1,
                        seed=40,
                        response_format={"type": "json_object"},
                    ).choices[0]
                    break
                except Exception as e:
                    sys.stderr.write(f"{type(e)}, {e}\n")
                    time.sleep(API_RETRY_SLEEP)
            t_completion = time.time()

            # Add the LLM's response to the messages array
            messages.append(completion.message.to_dict())
            
            # Save the messages to messages.json
            if not args.no_logging:
                with open(f"{save_dir}/messages.json", "w") as f:
                    json.dump(messages, f, indent=4)
            out = json.loads(completion.message.content)
        else:
            out = {"name": archive[i]["name"], "prompt": archive[i]["prompt"], "thought": archive[i].get("thought", "")}

        # VALIDATE PROMPT
        valid, error = validate_prompt(out, original_concepts)
        if not valid:
            next_prompt = (
                f"Prompt not valid. Error:\n{error}\nPlease generate the next one."
            )
            messages.append({"role": "user", "content": next_prompt})
            fitness = -1
            sys.stderr.write("PROMPT NOT VALID\n")
            continue
            
        # Handle the action
        try:
            action = Action(out["action"])
            if action == Action.ADD_NO_CONCEPT:
                concept_pool.add_no_concept(i)
            elif action == Action.ADD_ONE_CONCEPT:
                if "new_concept" not in out:
                    raise ValueError("Missing new_concept for ADD_ONE_CONCEPT action")
                concept_pool.add_concept(out["new_concept"], i)
            elif action == Action.ADD_MULTIPLE_CONCEPTS:
                if "new_concepts" not in out or not isinstance(out["new_concepts"], list):
                    raise ValueError("Missing or invalid new_concepts for ADD_MULTIPLE_CONCEPTS action")
                concept_pool.add_multiple_concepts(out["new_concepts"], i)
            else:
                raise ValueError(f"Invalid action: {action}")
            
            # Now perform mandatory recombination
            # Check if this recombination improved fitness compared to previous generation
            improved_fitness = False
            if i > 0 and len(fitness_history) > 0:
                previous_fitness = fitness_history[-1]
                improved_fitness = fitness > previous_fitness
            concept_pool.recombine_concepts(out["concepts_used"], i, improved_fitness)
        except Exception as e:
            sys.stderr.write(f"Error handling action: {e}\n")
            next_prompt = f"Action handling failed. Error:\n{str(e)}\nPlease generate the next one."
            messages.append({"role": "user", "content": next_prompt})
            fitness = -1
            continue
        
        sys.stderr.write(f"Generated artwork idea: {out['name']}\n")
        sys.stderr.write(f"Action taken: {action.value}\n")
        t_gen_start = time.time()

        # GENERATE ARTWORK
        sys.stderr.write("Generating artwork...\n")
        artwork_path = generate_artwork(
            prompt=out["prompt"],
            name=out["name"],
            output_dir=config["IMAGES_DIR"] or save_dir,  # Use images dir if available
            client=client,
            suffix=f"_gen{i}"
        )
        
        if not artwork_path:
            next_prompt = (
                f"Artwork generation failed. Please generate the next one with a different prompt."
            )
            messages.append({"role": "user", "content": next_prompt})
            fitness = -1
            sys.stderr.write("FAILED ARTWORK GENERATION\n")
            continue
        t_gen_end = time.time()
        sys.stderr.write(f"Artwork generated and saved to: {artwork_path}\n")

        # EVALUATE ARTWORK
        sys.stderr.write("Evaluating artwork...\n")
        evaluated, scores = evaluate_fitness(
            artwork_path, 
            out.get("concepts_used", []),  # Pass the concepts used in this generation
            concept_pool.concept_space_model  # Pass the concept model for similarity calculation
        )
        if not evaluated:
            next_prompt = (
                f"Evaluation failed. Error:\n{scores}\nPlease generate the next one."
            )
            messages.append({"role": "user", "content": next_prompt})
            fitness = -1
            sys.stderr.write("FAILED EVAL\n")
            continue
        
        # Unpack scores
        fitness, aesthetic_score, originality_score, diversity_score = scores
        
        t_eval_end = time.time()
        sys.stderr.write(f"Artwork evaluation complete. Fitness: {fitness:.2f} (Aesthetic: {aesthetic_score:.2f}, Originality: {originality_score:.2f}, Diversity: {diversity_score:.2f})\n")
        
        # Determine if fitness improved compared to previous generation
        improved_fitness = False
        if i > 0 and len(fitness_history) > 0:
            previous_fitness = fitness_history[-1]
            improved_fitness = fitness > previous_fitness
        
        # Record this combination in the concept memory
        concept_memory.record_combination(
            out.get("concepts_used", []), 
            fitness, 
            aesthetic_score, 
            originality_score, 
            diversity_score,  # Pass the diversity score
            i,  # generation number
            improved_fitness  # whether this combination improved fitness
        )
        
        # Add scores to history
        fitness_history.append(fitness)
        aesthetic_history.append(aesthetic_score)
        originality_history.append(originality_score)
        diversity_history.append(diversity_score)  # Store diversity score
        
        # Timing information
        prompt_time = t_completion - t_start
        artwork_time = t_gen_end - t_gen_start
        eval_time = t_eval_end - t_gen_end
        total_time = t_eval_end - t_start
        
        sys.stderr.write(f"Times - Prompt: {prompt_time:.2f}s, Artwork: {artwork_time:.2f}s, Eval: {eval_time:.2f}s, Total: {total_time:.2f}s\n")

        # Determine if fitness increased or decreased compared to previous generation
        fitness_change = ""
        if i > 0:
            if len(fitness_history) > 1:
                previous_fitness = fitness_history[-2]  # Get the previous generation's fitness                
                if fitness > previous_fitness:
                    fitness_change = "(Increased)"
                elif fitness < previous_fitness:
                    fitness_change = "(Decreased)"
                else:
                    fitness_change = "(No change)"
        
        # Save artwork info
        if not args.no_logging:
            artwork_info = {
                "generation": i,
                "name": out["name"],
                "prompt": out["prompt"],
                "thought": out["thought"],
                "action": action.value,
                "concepts_used": out.get("concepts_used", []),
                "new_concept": out.get("new_concept", None),
                "new_concepts": out.get("new_concepts", []),
                "fitness": fitness,
                "aesthetic_score": aesthetic_score,
                "originality_score": originality_score,
                "diversity_score": diversity_score,
                "path": os.path.relpath(artwork_path, save_dir),
                "timing": {
                    "prompt_time": prompt_time,
                    "artwork_time": artwork_time,
                    "eval_time": eval_time,
                    "total_time": total_time
                }
            }
            with open(f"{config['INFO_DIR']}/artwork_info_{i}.json", "w") as f:
                json.dump(artwork_info, f, indent=4)
            
            # Save concept pool state
            with open(f"{save_dir}/concept_pool.json", "w") as f:
                json.dump(concept_pool.to_dict(), f, indent=4)
            
            # Save concept memory state
            with open(f"{save_dir}/concept_memory.json", "w") as f:
                json.dump(concept_memory.to_dict(), f, indent=4)
            
            # Generate and save concept space visualization
            if len(concept_pool.get_concepts()) >= 2:
                concept_pool.visualize_concept_space(f"{config['INFO_DIR']}/concept_space_gen{i}.png", current_generation=i)
    
    # After all generations, plot the fitness history
    if not args.no_logging and len(fitness_history) > 0:
        plot_fitness_history(fitness_history, aesthetic_history, originality_history, diversity_history, config["INFO_DIR"])
        
        # Generate phylogenetic tree visualization
        plot_phylogenetic_tree(concept_pool, fitness_history, config["INFO_DIR"])
        
        # Generate final concept similarity evolution visualization
        if len(concept_pool.similarity_history) > 1:
            concept_pool.generate_visualizations(config["INFO_DIR"])
            
        # Generate concept-artwork relationship graph
        # First collect all artwork info from saved JSON files
        artwork_info_list = []
        for i in range(len(fitness_history)):
            try:
                with open(f"{config['INFO_DIR']}/artwork_info_{i}.json", "r") as f:
                    artwork_info = json.load(f)
                    artwork_info_list.append(artwork_info)
            except FileNotFoundError:
                sys.stderr.write(f"Warning: Could not find artwork info for generation {i}\n")
                
        # Generate the graph if we have artwork info
        if artwork_info_list:
            plot_concept_artwork_graph(concept_pool.get_history(), artwork_info_list, config["INFO_DIR"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-generations", type=int, default=5)
    parser.add_argument("--no-logging", action="store_true", default=False)
    parser.add_argument("--do-baselines", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--concepts", type=str, nargs="+", default=["romanticism"],
                        help="List of concepts to start with")
    parser.add_argument("--embedding-model", type=str, choices=["openai", "clip"], default="openai",
                        help="Model to use for concept embeddings (default: openai)")
    parser.add_argument("--no-historical-information", action="store_true", default=False,
                        help="Disable historical performance information in prompts to encourage exploration")
    parser.add_argument("--max-unsuccessful-uses", type=int, default=3,
                        help="Number of unsuccessful uses before a concept expires (default: 3)")
    parser.add_argument("--fitness-function", type=str, choices=["combined", "aesthetic", "originality", "diversity"], 
                        default="combined", help="Fitness function to use for evaluation (default: combined)")
    
    args = parser.parse_args()
    run_evolution(args)

if __name__ == "__main__":
    main() 