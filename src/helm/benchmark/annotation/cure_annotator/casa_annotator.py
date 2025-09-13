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
    template_path = os.path.join(os.path.dirname(__file__), "casa_prompt_templates.yaml")
    with open(template_path, 'r') as file:
        return yaml.safe_load(file)

PROMPT_TEMPLATES = load_prompt_templates()


class ThickCultureAgreementAnnotator(Annotator):
    def __init__(
        self,
        auto_client: AutoClient,
        models: Optional[list] = None,  # List of model names
        model_deployment: Optional[str] = None,
        metric: str = "agreement",
    ):
        self._auto_client = auto_client
        self.models = models or ["openai/gpt-4o-mini-2024-07-18"]
        self._model_deployment = model_deployment
        self.metric = metric
        self.name = f"thick_culture_autograder_{self.metric}"  # <-- Make name unique per metric

    def annotate(self, request_state: RequestState) -> Any:
        # Extract user query from the correct reference (index 2 based on scenario state)
        user_query = request_state.instance.references[1].output.text
        # Extract model response
        reasoning = request_state.result.completions[0].text
        # Use the correct variable names that match the template
        prompt = PROMPT_TEMPLATES[self.metric].format(user_query=user_query, reasoning=reasoning)
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
            result[self.metric] = "YES"
        elif "NO" in clean_output:
            result[self.metric] = "NO"
        else:
            result[self.metric] = output.strip()  # Return original if unclear
        return result



class ThickCultureConnotationAnnotator(Annotator):
    def __init__(
        self,
        auto_client: AutoClient,
        models: Optional[list] = None,  # List of model names
        model_deployment: Optional[str] = None,
        metric: str = "connotation",
    ):
        self._auto_client = auto_client
        self.models = models or ["openai/gpt-4o-mini-2024-07-18"]
        self._model_deployment = model_deployment
        self.metric = metric
        self.name = f"thick_culture_autograder_{self.metric}"  # <-- Make name unique per metric

    def annotate(self, request_state: RequestState) -> Any:
        symbol = request_state.instance.references[0].output.text
        reasoning = request_state.result.completions[0].text
        prompt = PROMPT_TEMPLATES[self.metric].format(symbol=symbol, reasoning=reasoning)
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


