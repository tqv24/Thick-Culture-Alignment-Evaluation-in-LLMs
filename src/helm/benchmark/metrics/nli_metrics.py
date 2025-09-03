from typing import List
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class NLIMetric(Metric):
    """
    Metric that evaluates NLI performance using annotator results.
    """
    
    def __init__(self):
        super().__init__()
    
    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations is not None
        
        # Get NLI annotation results
        nli_annotation = request_state.annotations.get("nli_annotator", None)
        
        if nli_annotation is None:
            raise ValueError("No NLI annotation found")
        
        # Extract scores from annotation
        if isinstance(nli_annotation, list) and len(nli_annotation) > 0:
            annotation_data = nli_annotation[0]
        else:
            annotation_data = nli_annotation
        
        entailment_score = annotation_data.get("entailment_score", 0.0)
        contradiction_score = annotation_data.get("contradiction_score", 0.0)
        neutral_score = annotation_data.get("neutral_score", 0.0)
        predicted_label = annotation_data.get("predicted_label", "")
        
        # Get reference label if available
        reference_label = None
        if request_state.instance.references:
            reference_label = request_state.instance.references[0].output.text.strip().upper()
        
        # Calculate exact match if reference is available
        exact_match = 0
        if reference_label and predicted_label:
            exact_match = 1 if predicted_label.upper() == reference_label else 0
        
        return [
            Stat(MetricName("nli_entailment_score")).add(entailment_score),
            Stat(MetricName("nli_contradiction_score")).add(contradiction_score),
            Stat(MetricName("nli_neutral_score")).add(neutral_score),
            Stat(MetricName("nli_exact_match")).add(exact_match),
        ]
