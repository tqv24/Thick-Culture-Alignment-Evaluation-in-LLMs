# CURE-HELM Benchmark: Cultural Undertsanding and Reasoning Evaluation 

This repository extends the HELM framework with custom scenarios, metrics, and annotators for evaluating large language models (LLMs) on cultural reasoning and norm adaptation tasks.

## Key Features

- **Rich Test Cases:**
  - Multiple cultural scenarios (CulturalBank, CASA, NormAD, and NCLB)
  - "Thin" (direct norm application) and "thick" (reasoning/explanation) variants
  - Multi-language and multi-country support
  - Easily extensible for new cultural or normative test cases

- **Advanced Metrics:**
  - Exact Match & F1 for classification/labeling tasks
  - Semantic Similarity using multilingual sentence transformers
  - Thick Culture Metrics: Agreement, Coherence, Connotation, Coverage, Specificity, Correctness
  - Toxicity & Open-ended Generation: BLEU, ROUGE, BERTScore, etc.

- **Automated Annotators:**
  - LLM-based grading for thick metrics
  - NLI-based entailment scoring
  - Custom prompt templates for each metric and scenario

## Usage Example

Run a test case with a specific scenario and metric:

```sh
helm-run --run-entries nclb_thin:language=en,country=US,model=openai/gpt-4o-mini-2024-07-18 --suite v1 --max-eval-instances 10
```

## Directory Structure

```
src/helm/benchmark/
  ├── annotation/
  ├── metrics/
  ├── run_specs/
  └── scenarios/
```

## Customization

- Add new scenarios in `scenarios/` and register in `run_specs/`.
- Add new metrics in `metrics/` and update the run spec.
- Add new annotators in `annotation/` and provide a prompt template YAML.





## Dataset Details

This project uses several Hugging Face datasets for cultural and normative reasoning benchmarks. Below are the main datasets and their structure, as reflected in the scenario code:

### Cultural Bank [[tqv25/culbank](https://huggingface.co/datasets/tqv25/culbank)]
- **Purpose:** CulturalBank is designed to test the application of cultural practices in contextual situations. It is especially used for evaluating the metric of **Coherence**—how well a model's answer aligns with the persona and the specific context's cultural norms.
- **Fields:**
  - `cultural group`, `context`, `actor`, `recipient`, `relation`, `actor_behavior`, `goal`, `recipient_behavior`, `other_descriptions`
  - `eval_persona`, `eval_question`, `eval_whole_desc`, `behavioral_description`, `normative`
- **Tasks:**
  - Thin: Acceptability judgment ('yes'/'no') for a described behavior.
  - Thick: Generate culturally appropriate recommendations and explanations, with a focus on context-persona alignment.

### CASA [[tqv25/casa](https://huggingface.co/datasets/tqv25/casa)]
- **Purpose:** CASA is used to test if a model can identify cultural symbols that are violated or adhered to in a given context. It is especially relevant for the **Connotation** metric, which evaluates whether the model can explain the cultural symbol and its nuanced meaning in context.
- **Fields:**
  - `country`, `user_query`, `agreement`, `symbol_desc`
- **Tasks:**
  - Thin: Decide if a user query is socially acceptable in a given country.
  - Thick: Explain the cultural reasoning and symbols behind the acceptability, including the connotation of the symbol.

### NormAd [[akhilayerukola/NormAd](https://huggingface.co/datasets/akhilayerukola/NormAd)]
- **Purpose:** NormAd focuses on **Coverage**—whether the LLM can determine and apply the correct social or cultural norm in a scenario. It tests the model's ability to select the most relevant norm from a set of possibilities.
- **Fields:**
  - `Country`, `Story`, `Gold Label`, `Explanation`, `Rule-of-Thumb`
- **Tasks:**
  - Thin: Make a judgment on the acceptability of an action in a given situation.
  - Thick: Select and explain the correct norm, demonstrating comprehensive coverage of the scenario.

### NCLB [[tqv25/nclb](https://huggingface.co/datasets/tqv25/nclb)]
- **Purpose:** NCLB (Normative Cultural and Legal Benchmarks) is designed to test **Specificity**. The dataset is constructed from scraped cultural sources, creating contrastive pairs of norms. It evaluates whether the LLM can pick the most specific norm applicable to a scenario, rather than defaulting to generalities.
- **Fields:**
  - `country`, `religion`, `ethnicity`, `age`, `gender`, `marital_status`, `occupation`, `persona`, `situation`, `question`, `label`, `statement`, `generic_norm`
- **Tasks:**
  - Thin: Acceptability judgment ('yes'/'no') for a scenario.
  - Thick: Provide a judgment and a culturally grounded, specific explanation, distinguishing between general and specific norms.




**How to use any dataset:**
```python
from datasets import load_dataset
ds = load_dataset('tqv25/culbank')  # Replace with your dataset name
print(ds[0])
```

