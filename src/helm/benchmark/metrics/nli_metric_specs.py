from typing import List
from helm.benchmark.metrics.metric import MetricSpec


def get_nli_metric_specs() -> List[MetricSpec]:
    """Get metric specs for NLI evaluation."""
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.nli_metrics.NLIMetric",
            args={},
        ),
    ]
