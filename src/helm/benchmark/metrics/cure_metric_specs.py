from typing import List

from helm.benchmark.metrics.metric import MetricSpec


def get_semantic_similarity_metric_specs(similarity_fn_name: str = "cosine") -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.cure_metrics.SemanticSimilarityMetric",
            args={"similarity_fn_name": similarity_fn_name},
        ),
    ]

def get_toxicity_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.cure_metrics.ToxicityMetric",
            args={},
        ),
    ]


def get_thick_culture_agreement_exact_match_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.cure_metrics.ThickCultureAgreementExactMatchMetric",
            args={"metric_name": "agreement"},
        ),
    ]


def get_thick_culture_coherence_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.cure_metrics.ThickCultureCoherenceMetric",
            args={"metric_name": "coherence"},
        ),
    ]


def get_thick_culture_connotation_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.cure_metrics.ThickCultureConnotationMetric",
            args={"metric_name": "connotation"},
        ),
    ]

def get_thick_culture_coverage_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.cure_metrics.ThickCultureCoverageMetric",
            args={"metric_name": "coverage"},
        ),
    ]

def get_thick_culture_specificity_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.cure_metrics.ThickCultureSpecificityMetric",
            args={},
        )
    ]



