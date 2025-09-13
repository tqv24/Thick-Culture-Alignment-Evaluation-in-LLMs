from typing import Any, Optional, Dict
import re
import yaml
import os

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.request import Request
from transformers import pipeline

# Load prompt templates from YAML file
def load_prompt_templates():
    template_path = os.path.join(os.path.dirname(__file__), "nclb_prompt_templates.yaml")
    with open(template_path, 'r') as file:
        return yaml.safe_load(file)

PROMPT_TEMPLATES = load_prompt_templates()



class ThickCultureAgreementAnnotator(Annotator):
    def __init__(
        self,
        auto_client: Optional[AutoClient] = None,  # Make optional since we don't need it
        models: Optional[list] = None,
        model_deployment: Optional[str] = None,
        metric: str = "agreement",
    ):
        # Don't store auto_client since we won't use it
        self.metric = metric
        self.name = f"thick_culture_autograder_{self.metric}"

    def annotate(self, request_state: RequestState) -> Any:
        # Extract the model's response
        reasoning = request_state.result.completions[0].text
        # Use regex to extract yes/no/neutral from the response
        extracted_answer = self._extract_answer(reasoning)
        return [self._postprocess(extracted_answer)]

    def _extract_answer(self, text: str) -> str:
        """Extract yes/no/neutral from the model response using regex."""
        # Clean the text and convert to lowercase
        clean_text = text.strip().lower()
        
        # Pattern 1: Look for **Answer**: format (which the model is using)
        answer_format_patterns = [
            r'\*\*answer\*\*:\s*(yes|no|neutral)',
            r'answer:\s*(yes|no|neutral)',
            r'\*\*answer\*\*:\s*\'(yes|no|neutral)\'',
            r'answer:\s*\'(yes|no|neutral)\'',
        ]
        
        for pattern in answer_format_patterns:
            match = re.search(pattern, clean_text)
            if match:
                return match.group(1)
        
        # Pattern 2: Look for yes/no/neutral at the beginning of the response
        start_patterns = [
            r'^\s*(yes|no|neutral)\b',
            r'^\s*\'(yes|no|neutral)\'\b',
            r'^\s*"(yes|no|neutral)"\b',
        ]
        
        for pattern in start_patterns:
            match = re.search(pattern, clean_text)
            if match:
                return match.group(1)
        
        # Pattern 3: Look for common answer formats anywhere in text
        general_patterns = [
            r'my answer is\s*(yes|no|neutral)',
            r'the answer is\s*(yes|no|neutral)',
            r'return.*?(yes|no|neutral)',
            r'\b(yes|no|neutral)\b.*appropriate',
            r'\b(yes|no|neutral)\b.*acceptable',
        ]
        
        for pattern in general_patterns:
            match = re.search(pattern, clean_text)
            if match:
                return match.group(1)
        
        # If no clear pattern found, return "EXTRACTION_FAILED" for debugging
        return "EXTRACTION_FAILED"

    def _postprocess(self, output: str) -> Dict[str, Any]:
        result = {}
        clean_output = output.strip().lower()
        
        if clean_output == "yes":
            result[self.metric] = "YES"
        elif clean_output == "no":
            result[self.metric] = "NO"
        elif clean_output == "neutral":
            result[self.metric] = "NEUTRAL"
        else:
            # If extraction failed, return the raw output for debugging
            result[self.metric] = output.strip()
        
        return result
    



class ThickCultureSpecificityAnnotator(Annotator):
    def __init__(
        self,
        auto_client: AutoClient,
        models: Optional[list] = None,  # List of model names
        model_deployment: Optional[str] = None,
        metric: str = "specificity",
    ):
        self._auto_client = auto_client
        self.models = models or ["openai/gpt-4o-mini-2024-07-18"]
        self._model_deployment = model_deployment
        self.metric = metric
        self.name = f"thick_culture_autograder_{self.metric}"  # <-- Make name unique per metric

    def annotate(self, request_state: RequestState) -> Any:
        norm = request_state.instance.references[0].output.text
        reasoning = request_state.result.completions[0].text
        prompt = PROMPT_TEMPLATES[self.metric].format(norm=norm, reasoning=reasoning)
        results = []
        for model in self.models:
            annotator_request = Request(
                model=model,
                model_deployment=self._model_deployment or model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=8,
            )
            annotator_response = self._auto_client.make_request(annotator_request)
            if not annotator_response.success:
                raise Exception(f"Annotation request failed: {annotator_response.error}")
            annotator_response_text = annotator_response.completions[0].text
            results.append(self._postprocess(annotator_response_text))
        return results

    def _postprocess(self, output: str) -> Dict[str, Any]:
        result = {}
        # Return the raw YES/NO response for exact match comparison
        clean_output = output.strip().upper()
        if "YES" in clean_output:
            result[self.metric] = 1
        elif "NO" in clean_output:
            result[self.metric] = 0
        else:
            result[self.metric] = 0 # Return original if unclear
        return result