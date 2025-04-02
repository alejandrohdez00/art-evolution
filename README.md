# Art-Evolution
We use an evolutionary approach to discover novel and interesting conceptual combinations in visual art. We use LLMs (GPT-4o) as the mutation operator, i.e. it will propose new concepts and recombine them with previous concepts to create novel artworks. In each generation, we evaluate the fitness of the artwork and give this feedback to the LLM to guide the search to novel artworks. This approach is inspired by [Discovering Preference Optimization Algorithms with and for Large Language Models](https://arxiv.org/abs/2406.08414) but adapted to the visual art domain and concept recombination.

The fitness function is a combination of the following metrics:
- Aesthetic score: *How aesthetically pleasing is the artwork?* We use the [Aesthetic Predictor model](https://github.com/discus0434/aesthetic-predictor-v2-5) to score the artwork.
- Originality: *How original is the artwork compared to WikiArt artworks?* We use a fine-tuned ResNet152 to extract features from the artwork and then we measure the closest match to the features of WikiArt artworks.
- Diversity: *How diverse are the concepts used in the artwork?* We compute the average cosine similarity between the concepts used in the artwork.

## Setup

Create a virtual environment and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Add your OpenAI API key to the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

## Run
To run the art evolution, use the following command:

```bash
python art_evolution/run_evolution.py --concepts mountain --num-generations 15
```

The `--concepts` argument specifies the set of concepts for which you want to discover novel combinations. These concepts are the original concepts and will be always used in the images.

*Note: Currently, DALL-E 3 is used to generate the images. We have seen that sometimes it generates bad images when the prompt is complex. We will add the new GPT-4o image generation as soon as it is available.*

