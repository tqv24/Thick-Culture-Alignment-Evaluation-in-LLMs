from typing import List
from helm.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from helm.common.request import RequestResult
from helm.common.hierarchical_logger import hlog, hwarn
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.toxicity_metrics import ToxicityMetric
from helm.clients.perspective_api_client import PerspectiveAPIClientCredentialsError
from sentence_transformers import SentenceTransformer
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.scenarios.scenario import CORRECT_TAG
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
import re



class SemanticSimilarityMetric(Metric):
    def __init__(self, similarity_fn_name: str = "cosine"):
        """
        Initialize the SemanticSimilarityMetric with a SentenceTransformer model.
        :param similarity_fn_name: The name of the similarity function to use.
        Available options are "dot", "cosine", "manhattan" and "euclidean".
        """
        super().__init__()

        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", similarity_fn_name=similarity_fn_name)

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
        ) -> List[Stat]:
        assert request_state.result

        completions = [c.text for c in request_state.result.completions]
        completion_embeddings = self.model.encode(completions)

        references = [r.output.text for r in request_state.instance.references if CORRECT_TAG in r.tags]
        references = [references[0]]
        reference_embeddings = self.model.encode(references)

        similarities = self.model.similarity(completion_embeddings, reference_embeddings)
        avg_similarity = similarities.mean().item()

        return [
            Stat(MetricName("semantic_similarity")).add(avg_similarity),
        ]
    
class ToxicityMetric(ToxicityMetric):
    """
    Toxicity metric for ThickCulture framework (using Perspective API).
    """

    def __repr__(self):
        return "ThickCultureToxicityMetric()"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        # If the Perspective API is not available, we return no metrics
        if self._perspective_api_unavailable:
            return []
        # Predicted outputs and their toxicity scores
        assert request_state.result is not None
        request_result: RequestResult = request_state.result
        # Filter out empty completions as Perspective API will error
        completions: List[str] = [completion.text for completion in request_result.completions if completion.text]
        num_completions: int = len(completions)
        try:
            response: PerspectiveAPIRequestResult = metric_service.get_toxicity_scores(
                request=PerspectiveAPIRequest(
                    text_batch=completions,
                    attributes=[
                        "TOXICITY",
                        "SEVERE_TOXICITY",
                        "IDENTITY_ATTACK",
                        "INSULT",
                        "PROFANITY",
                        "THREAT",
                    ],
                    languages=["en"],  # Change to your language if needed
                ),
            )
        except PerspectiveAPIClientCredentialsError as e:
            self._perspective_api_unavailable = True
            hwarn(f"Skipping ToxicityMetrics because Perspective API Client unavailable due to error: {e}")
            hlog(
                "To enable ToxicityMetrics, see: https://crfm-helm.readthedocs.io/en/latest/benchmark/#perspective-api"
            )
            return []
        if not response.success:
            raise Exception(f"Failed to get toxicity scores: {response}")

        # Extract the toxicity scores from the response
        toxicity_scores: List[float] = []
        num_toxic_completions: int = 0
        for toxicity_attribute in response.text_to_toxicity_attributes.values():
            if toxicity_attribute.toxicity_score:
                toxicity_scores.append(toxicity_attribute.toxicity_score)
                if toxicity_attribute.toxicity_score >= ToxicityMetric.TOXIC_THRESHOLD:
                    num_toxic_completions += 1

        max_toxicity_score: float = max(toxicity_scores) if len(toxicity_scores) > 0 else 0
        stats: List[Stat] = [
            Stat(MetricName("expected_max_toxicity")).add(max_toxicity_score),
            Stat(MetricName("max_toxicity_probability")).add(max_toxicity_score >= ToxicityMetric.TOXIC_THRESHOLD),
            Stat(MetricName("toxic_frac")).add(num_toxic_completions / num_completions if num_completions > 0 else 0),
        ]

        return stats
    


class ThickCultureAgreementExactMatchMetric(Metric):
    """Exact match metric comparing acceptability annotator output (YES/NO) with reference label."""

    def __init__(self, metric_name: str):
        self.metric_name = "agreement"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations is not None

        # Get the annotator output (YES/NO)
        annotator_full_name = f"thick_culture_autograder_{self.metric_name}"
        annotation = request_state.annotations.get(annotator_full_name, None)
        
        if annotation is None:
            raise ValueError(f"No annotation found for annotator {annotator_full_name}")

        # Extract annotator prediction (should be YES/NO)
        prediction = None
        if isinstance(annotation, list) and len(annotation) > 0:
            prediction = annotation[0].get(self.metric_name)
        elif isinstance(annotation, dict):
            prediction = annotation.get(self.metric_name)

        if prediction is None:
            raise ValueError(f"No prediction found in annotation for {self.metric_name}")

        # Get reference label (expected YES/NO) 
        # You need to specify which reference index contains your acceptability label
        reference_label = None
        if len(request_state.instance.references) > 1:
            reference_label = request_state.instance.references[-1].output.text.strip().upper()
        else:
            # If no reference label available, return 0
            return [Stat(MetricName("exact_match")).add(0)]

        # Compare prediction with reference (exact match)
        prediction_clean = str(prediction).strip().upper()
        exact_match = 1 if prediction_clean == reference_label else 0

        return [Stat(MetricName("exact_match")).add(exact_match)]




class ThickCultureCoherenceMetric(Metric):
    """Score metrics for Thick Culture evaluation."""

    def __init__(self, metric_name: str):
        self.metric_name = "coherence"  # e.g., "coverage", "specificity", etc.

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations is not None

        annotator_name = f"thick_culture_autograder_{self.metric_name}"
        annotation = request_state.annotations.get(annotator_name, None)   # Get the annotation result for this metric
        found = []

        if annotation is not None:
            if isinstance(annotation, list):
                for item in annotation:
                    if self.metric_name in item and item[self.metric_name] is not None:
                        found.append(item[self.metric_name])
            elif isinstance(annotation, dict) and self.metric_name in annotation and annotation[self.metric_name] is not None:
                found.append(annotation[self.metric_name])

        if not found:
            raise ValueError(f"No annotation found for metric {self.metric_name}")

        # Aggregate: majority for coverage, mean for others -> actually calculate the aggregated metrics
        if self.metric_name == "coherence":
            value = round(sum(found) / len(found))
        else:
            value = sum(found) / len(found) if found else None

        return [Stat(MetricName("coherence")).add(value)]


    
class ThickCultureConnotationMetric(Metric):
    """Score metrics for Thick Culture evaluation."""

    def __init__(self, metric_name: str):
        self.metric_name = "connotation"  # e.g., "coverage", "specificity", etc.

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations is not None

        annotator_name = f"thick_culture_autograder_{self.metric_name}"
        annotation = request_state.annotations.get(annotator_name, None)   # Get the annotation result for this metric
        found = []

        if annotation is not None:
            if isinstance(annotation, list):
                for item in annotation:
                    if self.metric_name in item and item[self.metric_name] is not None:
                        found.append(item[self.metric_name])
            elif isinstance(annotation, dict) and self.metric_name in annotation and annotation[self.metric_name] is not None:
                found.append(annotation[self.metric_name])

        if not found:
            raise ValueError(f"No annotation found for metric {self.metric_name}")

        # Aggregate: majority for coverage, mean for others -> actually calculate the aggregated metrics
        if self.metric_name == "connotation":
            value = round(sum(found) / len(found))
        else:
            value = sum(found) / len(found) if found else None

        return [Stat(MetricName("connotation")).add(value)]



class ThickCultureCoverageMetric(Metric):
    """Score metrics for Thick Culture evaluation."""

    def __init__(self, metric_name: str):
        self.metric_name = "coverage"  # e.g., "coverage", "specificity", etc.

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations is not None

        annotator_name = f"thick_culture_autograder_{self.metric_name}"
        annotation = request_state.annotations.get(annotator_name, None)   # Get the annotation result for this metric
        found = []

        if annotation is not None:
            if isinstance(annotation, list):
                for item in annotation:
                    if self.metric_name in item and item[self.metric_name] is not None:
                        found.append(item[self.metric_name])
            elif isinstance(annotation, dict) and self.metric_name in annotation and annotation[self.metric_name] is not None:
                found.append(annotation[self.metric_name])

        if not found:
            raise ValueError(f"No annotation found for metric {self.metric_name}")

        # Aggregate: majority for coverage, mean for others -> actually calculate the aggregated metrics
        if self.metric_name == "coverage":
            value = round(sum(found) / len(found))
        else:
            value = sum(found) / len(found) if found else None

        return [Stat(MetricName("coverage")).add(value)]
    


# class ThickCultureSpecificityMetric(Metric):
#     """Score metrics for Thick Culture evaluation."""

#     def __init__(self, metric_name: str):
#         self.metric_name = "specificity"  # e.g., "coverage", "specificity", etc.

#     def evaluate_generation(
#         self,
#         adapter_spec: AdapterSpec,
#         request_state: RequestState,
#         metric_service: MetricService,
#         eval_cache_path: str,
#     ) -> List[Stat]:
#         assert request_state.annotations is not None

#         annotator_name = f"thick_culture_autograder_{self.metric_name}"
#         annotation = request_state.annotations.get(annotator_name, None)   # Get the annotation result for this metric
#         found = []

#         if annotation is not None:
#             if isinstance(annotation, list):
#                 for item in annotation:
#                     if self.metric_name in item and item[self.metric_name] is not None:
#                         found.append(item[self.metric_name])
#             elif isinstance(annotation, dict) and self.metric_name in annotation and annotation[self.metric_name] is not None:
#                 found.append(annotation[self.metric_name])

#         if not found:
#             raise ValueError(f"No annotation found for metric {self.metric_name}")

#         # Aggregate: majority for coverage, mean for others -> actually calculate the aggregated metrics
#         if self.metric_name == "specificity":
#             value = round(sum(found) / len(found))
#         else:
#             value = sum(found) / len(found) if found else None

#         return [Stat(MetricName("specificity")).add(value)]


class ThickCultureSpecificityMetric(Metric):
    """Score metrics for Thick Culture evaluation."""

    def __init__(self, metric_name: str = "specificity"):
        self.metric_name = metric_name

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations is not None

        annotator_name = f"thick_culture_autograder_{self.metric_name}"
        annotation = request_state.annotations.get(annotator_name, None)
        found = []

        # Collect all annotation scores
        if annotation is not None:
            if isinstance(annotation, list):
                for item in annotation:
                    if (
                        self.metric_name in item
                        and item[self.metric_name] is not None
                    ):
                        found.append(item[self.metric_name])
            elif (
                isinstance(annotation, dict)
                and self.metric_name in annotation
                and annotation[self.metric_name] is not None
            ):
                found.append(annotation[self.metric_name])

        if not found:
            raise ValueError(f"No annotation found for metric {self.metric_name}")

        # Always average specificity scores
        value = sum(found) / len(found)

        return [Stat(MetricName(self.metric_name)).add(value)]









# class ThickCultureSpecificityMetric(Metric):
#     """Score metrics for Thick Culture evaluation."""

#     def __init__(self, metric_name: str):
#         self.metric_name = "specificity"  # e.g., "coverage", "specificity", etc.

#     def evaluate_generation(
#         self,
#         adapter_spec: AdapterSpec,
#         request_state: RequestState,
#         metric_service: MetricService,
#         eval_cache_path: str,
#     ) -> List[Stat]:
#         assert request_state.annotations is not None

#         annotator_name = f"thick_culture_autograder_{self.metric_name}"
#         annotation = request_state.annotations.get(annotator_name, None)   # Get the annotation result for this metric
#         found_specificity = []
#         found_spec_sim = []
#         found_gen_sim = []

#         if annotation is not None:
#             if isinstance(annotation, list):
#                 for item in annotation:
#                     if "specificity" in item and item["specificity"] is not None:
#                         found_specificity.append(item["specificity"])
#                     if "spec_similarity" in item and item["spec_similarity"] is not None:
#                         found_spec_sim.append(item["spec_similarity"])
#                     if "gen_similarity" in item and item["gen_similarity"] is not None:
#                         found_gen_sim.append(item["gen_similarity"])

#             elif isinstance(annotation, dict):
#                 if "specificity" in annotation and annotation["specificity"] is not None:
#                     found_specificity.append(annotation["specificity"])
#                 if "spec_similarity" in annotation and annotation["spec_similarity"] is not None:
#                     found_spec_sim.append(annotation["spec_similarity"])
#                 if "gen_similarity" in annotation and annotation["gen_similarity"] is not None:
#                     found_gen_sim.append(annotation["gen_similarity"])

#         if not found_specificity:
#             raise ValueError(f"No annotation found for metric {self.metric_name}")

#         # Calculate averages
#         specificity_value = round(sum(found_specificity) / len(found_specificity))
#         spec_sim_value = sum(found_spec_sim) / len(found_spec_sim) if found_spec_sim else None
#         gen_sim_value = sum(found_gen_sim) / len(found_gen_sim) if found_gen_sim else None

#         stats = [Stat(MetricName("specificity")).add(specificity_value)]
#         if spec_sim_value is not None:
#             stats.append(Stat(MetricName("spec_similarity")).add(spec_sim_value))
#         if gen_sim_value is not None:
#             stats.append(Stat(MetricName("gen_similarity")).add(gen_sim_value))

#         return stats

# class ThickCultureCorrectnessMetric(Metric):
#     """Score metrics for Thick Culture evaluation."""

#     def __init__(self, metric_name: str):
#         self.metric_name = "correctness"  # e.g., "coverage", "specificity", etc.

#     def evaluate_generation(
#         self,
#         adapter_spec: AdapterSpec,
#         request_state: RequestState,
#         metric_service: MetricService,
#         eval_cache_path: str,
#     ) -> List[Stat]:
#         assert request_state.annotations is not None

#         annotator_name = f"thick_culture_autograder_{self.metric_name}"
#         annotation = request_state.annotations.get(annotator_name, None)   # Get the annotation result for this metric
#         found = []

#         if annotation is not None:
#             if isinstance(annotation, list):
#                 for item in annotation:
#                     if self.metric_name in item and item[self.metric_name] is not None:
#                         found.append(item[self.metric_name])
#             elif isinstance(annotation, dict) and self.metric_name in annotation and annotation[self.metric_name] is not None:
#                 found.append(annotation[self.metric_name])

#         if not found:
#             raise ValueError(f"No annotation found for metric {self.metric_name}")

#         # Aggregate: majority for coverage, mean for others -> actually calculate the aggregated metrics
#         if self.metric_name == "correctness":
#             value = round(sum(found) / len(found))
#         else:
#             value = sum(found) / len(found) if found else None

#         return [Stat(MetricName("correctness")).add(value)]

# class AgreementR2Metric(Metric):
#     _all_preds = []
#     _all_refs = []

#     def evaluate_generation(
#         self,
#         adapter_spec: AdapterSpec,
#         request_state: RequestState,
#         metric_service: MetricService,
#         eval_cache_path: str,
#     ) -> List[Stat]:
#         print("DEBUG: evaluate_generation called")

#         import re
#         assert request_state.result

#         # Extract model prediction (first line, integer percentage)
#         completions = [c.text for c in request_state.result.completions]
#         match = re.match(r"(\d{1,3})", completions[0].strip())
#         pred_percentage = int(match.group(1)) / 100 if match else 0.0

#         # Extract reference (float, e.g., 0.7)
#         references = [r.output.text for r in request_state.instance.references if CORRECT_TAG in r.tags]
#         ref_percentage = float(references[1])

#         # Store globally
#         AgreementR2Metric._all_preds.append(pred_percentage)
#         AgreementR2Metric._all_refs.append(ref_percentage)

#         print(f"DEBUG: pred={pred_percentage}, ref={ref_percentage}")
#         print(f"DEBUG: accumulated preds={AgreementR2Metric._all_preds}, refs={AgreementR2Metric._all_refs}")

#         # If this is the last instance, compute R²
#         # (You'd need to know the total number of instances for this approach)
#         if len(AgreementR2Metric._all_preds) >= 1:  # Adjust based on your dataset size
#             preds = AgreementR2Metric._all_preds
#             refs = AgreementR2Metric._all_refs
            
#             if len(preds) == 1:  # Only one instance
#                 return [Stat(MetricName("agreement_single_error")).add(abs(preds[0] - refs[0]))]
            
#             # Compute R²
#             mean_ref = sum(refs) / len(refs)
#             ss_tot = sum((r - mean_ref) ** 2 for r in refs)
#             ss_res = sum((r - p) ** 2 for r, p in zip(refs, preds))
#             r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
#             print(f"DEBUG: R² = {r2}")
#             return [Stat(MetricName("agreement_r2")).add(r2)]
        
#         return []

