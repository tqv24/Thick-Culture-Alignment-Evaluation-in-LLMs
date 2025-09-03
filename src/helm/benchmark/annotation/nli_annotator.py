from typing import Any, Dict
from transformers import pipeline
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator


class NLIAnnotator(Annotator):
    """
    Annotator that uses a pre-trained NLI model to classify entailment.
    """
    
    def __init__(self, model_name: str = "microsoft/deberta-large-mnli"):
        self.nli_pipeline = pipeline(
            "text-classification", 
            model=model_name,
            return_all_scores=True
        )
        self.name = "nli_annotator"
    
    def annotate(self, request_state: RequestState) -> Any:
        # Get premise and hypothesis
        # Assuming premise is in the input and hypothesis is the model's completion
        premise = request_state.instance.input.text
        hypothesis = request_state.result.completions[0].text
        
        # Create NLI input
        nli_input = f"{premise} [SEP] {hypothesis}"
        
        # Get NLI prediction
        results = self.nli_pipeline(nli_input)
        
        # Extract scores for each label
        scores = {result['label']: result['score'] for result in results[0]}
        
        return [{
            "entailment_score": scores.get("ENTAILMENT", 0.0),
            "contradiction_score": scores.get("CONTRADICTION", 0.0),
            "neutral_score": scores.get("NEUTRAL", 0.0),
            "predicted_label": max(scores, key=scores.get)
        }]
