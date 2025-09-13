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
    template_path = os.path.join(os.path.dirname(__file__), "culturalbank_prompt_templates.yaml")
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
        behavior_desc = request_state.instance.references[-2].output.text
        reasoning = request_state.result.completions[0].text
        prompt = PROMPT_TEMPLATES[self.metric].format(behavior_desc=behavior_desc, reasoning=reasoning)
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


class ThickCultureCoherenceAnnotator(Annotator):
    def __init__(
        self,
        auto_client: AutoClient,
        models: Optional[list] = None,  # List of model names
        model_deployment: Optional[str] = None,
        metric: str = "coherence",
    ):
        self._auto_client = auto_client
        self.models = models or ["openai/gpt-4o-mini-2024-07-18"]
        self._model_deployment = model_deployment
        self.metric = metric
        self.name = f"thick_culture_autograder_{self.metric}"  # <-- Make name unique per metric

    def annotate(self, request_state: RequestState) -> Any:
        persona = request_state.instance.references[1].output.text
        context = request_state.instance.references[2].output.text
        explanation = request_state.instance.references[0].output.text
        reasoning = request_state.result.completions[0].text
        prompt = PROMPT_TEMPLATES[self.metric].format(persona=persona, context=context, explanation=explanation, reasoning=reasoning)
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
    





# class ThickCultureCorrectnessAnnotator(Annotator):
#     def __init__(
#         self,
#         auto_client: AutoClient,
#         models: Optional[list] = None,  # List of model names
#         model_deployment: Optional[str] = None,
#         metric: str = "correctness",
#     ):
#         self._auto_client = auto_client
#         self.models = models or ["openai/gpt-4o-mini-2024-07-18"]
#         self._model_deployment = model_deployment
#         self.metric = metric
#         self.name = f"thick_culture_autograder_{self.metric}" 

#     def annotate(self, request_state: RequestState) -> Any:
#         cultural_knowledge = request_state.instance.references[0].output.text
#         label = request_state.instance.references[-1].output.text
#         reasoning = request_state.result.completions[0].text
#         prompt = PROMPT_TEMPLATES[self.metric].format(cultural_knowledge=cultural_knowledge, label=label, reasoning=reasoning)

#         results = []
#         for model in self.models:
#             annotator_request = Request(
#                 model=model,
#                 model_deployment=self._model_deployment or model,
#                 prompt=prompt,
#                 temperature=0.0,
#                 max_tokens=8,
#             )
#             annotator_response = self._auto_client.make_request(annotator_request)
#             if not annotator_response.success:
#                 raise Exception(f"Annotation request failed: {annotator_response.error}")
#             annotator_response_text = annotator_response.completions[0].text
#             results.append(self._postprocess(annotator_response_text))
#         return results
    
#     def _postprocess(self, output: str) -> Dict[str, Any]:
#         result = {}
#         # Return the raw YES/NO response for exact match comparison
#         clean_output = output.strip().upper()
#         if "YES" in clean_output:
#             result[self.metric] = 1
#         elif "NO" in clean_output:
#             result[self.metric] = 0
#         else:
#             result[self.metric] = 0 # Return original if unclear
#         return result
    






# class ThickCultureCoverageAnnotator(Annotator):
#     def __init__(
#         self,
#         auto_client: AutoClient,
#         models: Optional[list] = None,  # List of model names
#         model_deployment: Optional[str] = None,
#         metric: str = "coverage",
#     ):
#         self._auto_client = auto_client
#         self.models = models or ["openai/gpt-4o-mini-2024-07-18"]
#         self._model_deployment = model_deployment
#         self.metric = metric
#         self.name = f"thick_culture_autograder_{self.metric}"  # <-- Make name unique per metric

#     def annotate(self, request_state: RequestState) -> Any:
#         reference = request_state.instance.references[1].output.text
#         reasoning = request_state.result.completions[0].text
#         prompt = PROMPT_TEMPLATES[self.metric].format(reference=reference, reasoning=reasoning)
#         results = []
#         for model in self.models:
#             annotator_request = Request(
#                 model=model,
#                 model_deployment=self._model_deployment or model,
#                 prompt=prompt,
#                 temperature=0.0,
#                 max_tokens=8,
#             )
#             annotator_response = self._auto_client.make_request(annotator_request)
#             if not annotator_response.success:
#                 raise Exception(f"Annotation request failed: {annotator_response.error}")
#             annotator_response_text = annotator_response.completions[0].text
#             results.append(self._postprocess(annotator_response_text))
#         return results

#     def _postprocess(self, output: str) -> Dict[str, Any]:
#         result = {}
#         result[self.metric] = 1 if "yes" in output.lower() else 0
#         return result
    



# class ThickCultureSpecificityAnnotator(Annotator):
#     def __init__(
#         self,
#         auto_client: AutoClient,
#         models: Optional[list] = None,  # List of model names
#         model_deployment: Optional[str] = None,
#         metric: str = "specificity",
#     ):
#         self._auto_client = auto_client
#         self.models = models or ["openai/gpt-4o-mini-2024-07-18"]
#         self._model_deployment = model_deployment
#         self.metric = metric
#         self.name = f"thick_culture_autograder_{self.metric}"  # <-- Make name unique per metric

#     def annotate(self, request_state: RequestState) -> Any:
#         reference = request_state.instance.references[2].output.text
#         reasoning = request_state.result.completions[0].text
#         prompt = PROMPT_TEMPLATES[self.metric].format(reference=reference, reasoning=reasoning)
#         results = []
#         for model in self.models:
#             annotator_request = Request(
#                 model=model,
#                 model_deployment=self._model_deployment or model,
#                 prompt=prompt,
#                 temperature=0.0,
#                 max_tokens=8,
#             )
#             annotator_response = self._auto_client.make_request(annotator_request)
#             if not annotator_response.success:
#                 raise Exception(f"Annotation request failed: {annotator_response.error}")
#             annotator_response_text = annotator_response.completions[0].text
#             results.append(self._postprocess(annotator_response_text))
#         return results

#     def _postprocess(self, output: str) -> Dict[str, Any]:
#         result = {}
#         result[self.metric] = 1 if "yes" in output.lower() else 0

#         return result


# class NLIAnnotator(Annotator):
#     """
#     Annotator that uses a pre-trained NLI model to classify entailment.
#     """
    
#     def __init__(self, model: str = "microsoft/deberta-large-mnli"):
#         self.model = model
#         self.nli_pipeline = pipeline(
#             "text-classification", 
#             model=model,
#             return_all_scores=True
#         )
#         self.name = "nli_annotator"
    
#     def annotate(self, request_state: RequestState) -> Any:
#         # Get premise and hypothesis
#         # Assuming premise is in the reference and hypothesis is the model's completion
#         premise = request_state.instance.references[0].output.text
#         hypothesis = request_state.result.completions[0].text
        
#         # Create NLI input
#         nli_input = f"{premise} [SEP] {hypothesis}"
        
#         # Get NLI prediction
#         results = self.nli_pipeline(nli_input)
        
#         # Extract scores for each label
#         scores = {result['label']: result['score'] for result in results[0]}
        
#         return [{
#             "entailment_score": scores.get("ENTAILMENT", 0.0),
#             "contradiction_score": scores.get("CONTRADICTION", 0.0),
#             "neutral_score": scores.get("NEUTRAL", 0.0),
#             "predicted_label": max(scores, key=scores.get)
#         }]
    # def _postprocess(self, output: str) -> Dict[str, Any]:
    #     result = {}
    #     # Return the raw YES/NO response for exact match comparison
    #     clean_output = output.strip().upper()
    #     if "YES" in clean_output:
    #         result[self.metric] = 1
    #     elif "NO" in clean_output:
    #         result[self.metric] = 0
    #     else:
    #         result[self.metric] = 0 # Return original if unclear
    #     return result
    

# class ThickCultureCoverageAnnotator(Annotator):
#     def __init__(
#         self,
#         auto_client: AutoClient,
#         models: Optional[list] = None,  # List of model names
#         model_deployment: Optional[str] = None,
#         metric: str = "coverage",
#     ):
#         self._auto_client = auto_client
#         self.models = models or ["openai/gpt-4o-mini-2024-07-18"]
#         self._model_deployment = model_deployment
#         self.metric = metric
#         self.name = f"thick_culture_autograder_{self.metric}"  # <-- Make name unique per metric

#     def annotate(self, request_state: RequestState) -> Any:
#         reference = request_state.instance.references[1].output.text
#         reasoning = request_state.result.completions[0].text
#         prompt = PROMPT_TEMPLATES[self.metric].format(reference=reference, reasoning=reasoning)
#         results = []
#         for model in self.models:
#             annotator_request = Request(
#                 model=model,
#                 model_deployment=self._model_deployment or model,
#                 prompt=prompt,
#                 temperature=0.0,
#                 max_tokens=8,
#             )
#             annotator_response = self._auto_client.make_request(annotator_request)
#             if not annotator_response.success:
#                 raise Exception(f"Annotation request failed: {annotator_response.error}")
#             annotator_response_text = annotator_response.completions[0].text
#             results.append(self._postprocess(annotator_response_text))
#         return results

#     def _postprocess(self, output: str) -> Dict[str, Any]:
#         result = {}
#         result[self.metric] = 1 if "yes" in output.lower() else 0
#         return result
    



# class ThickCultureSpecificityAnnotator(Annotator):
#     def __init__(
#         self,
#         auto_client: AutoClient,
#         models: Optional[list] = None,  # List of model names
#         model_deployment: Optional[str] = None,
#         metric: str = "specificity",
#     ):
#         self._auto_client = auto_client
#         self.models = models or ["openai/gpt-4o-mini-2024-07-18"]
#         self._model_deployment = model_deployment
#         self.metric = metric
#         self.name = f"thick_culture_autograder_{self.metric}"  # <-- Make name unique per metric

#     def annotate(self, request_state: RequestState) -> Any:
#         reference = request_state.instance.references[2].output.text
#         reasoning = request_state.result.completions[0].text
#         prompt = PROMPT_TEMPLATES[self.metric].format(reference=reference, reasoning=reasoning)
#         results = []
#         for model in self.models:
#             annotator_request = Request(
#                 model=model,
#                 model_deployment=self._model_deployment or model,
#                 prompt=prompt,
#                 temperature=0.0,
#                 max_tokens=8,
#             )
#             annotator_response = self._auto_client.make_request(annotator_request)
#             if not annotator_response.success:
#                 raise Exception(f"Annotation request failed: {annotator_response.error}")
#             annotator_response_text = annotator_response.completions[0].text
#             results.append(self._postprocess(annotator_response_text))
#         return results

#     def _postprocess(self, output: str) -> Dict[str, Any]:
#         result = {}
#         result[self.metric] = 1 if "yes" in output.lower() else 0

#         return result


# class NLIAnnotator(Annotator):
#     """
#     Annotator that uses a pre-trained NLI model to classify entailment.
#     """
    
#     def __init__(self, model: str = "microsoft/deberta-large-mnli"):
#         self.model = model
#         self.nli_pipeline = pipeline(
#             "text-classification", 
#             model=model,
#             return_all_scores=True
#         )
#         self.name = "nli_annotator"
    
#     def annotate(self, request_state: RequestState) -> Any:
#         # Get premise and hypothesis
#         # Assuming premise is in the reference and hypothesis is the model's completion
#         premise = request_state.instance.references[0].output.text
#         hypothesis = request_state.result.completions[0].text
        
#         # Create NLI input
#         nli_input = f"{premise} [SEP] {hypothesis}"
        
#         # Get NLI prediction
#         results = self.nli_pipeline(nli_input)
        
#         # Extract scores for each label
#         scores = {result['label']: result['score'] for result in results[0]}
        
#         return [{
#             "entailment_score": scores.get("ENTAILMENT", 0.0),
#             "contradiction_score": scores.get("CONTRADICTION", 0.0),
#             "neutral_score": scores.get("NEUTRAL", 0.0),
#             "predicted_label": max(scores, key=scores.get)
#         }]

