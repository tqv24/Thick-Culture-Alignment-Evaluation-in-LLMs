from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import (get_exact_match_metric_specs,
                                                        get_f1_metric_specs,
                                                        get_open_ended_generation_metric_specs)
from helm.benchmark.metrics.cure_metric_specs import (  get_semantic_similarity_metric_specs,
                                                        get_thick_culture_agreement_exact_match_metric_specs,
                                                        get_thick_culture_coherence_metric_specs,
                                                        get_thick_culture_connotation_metric_specs,
                                                        get_thick_culture_coverage_metric_specs,
                                                        get_thick_culture_specificity_metric_specs,
                                                    )
from helm.benchmark.metrics.cure_metric_specs import get_toxicity_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.annotation.annotator import AnnotatorSpec

INSTRUCTIONS = {

    "normad_thin": {
        "en": {
            "instructions": "",
            "input_noun": "",
            "output_noun": "",
    },
    },
    "normad_thick": {
        "en": {
            "instructions": "",
            "input_noun": "",
            "output_noun": "",
        },
    },
}

COUNTRIES = {
    "US": "United States",
    # "VN": "Vietnam",
}


@run_spec_function("culturalbank_thin")
def get_apply_norms_culturalbank_spec(language: str, country: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.cure_scenarios.CulturalBankThinScenario",
        args={
            "language": language,
        },
    )

    adapter_spec = get_generation_adapter_spec(
        max_tokens=5,
        num_outputs=5,
        stop_sequences=[],
    )
    return RunSpec(
        name="culturalbank_thin",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_f1_metric_specs(),
        groups=["cure", "culturalbank_thin"],
    )

@run_spec_function("culturalbank_thick")
def get_reasoning_norms_culturalbank_spec(language: str, country: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.cure_scenarios.CulturalBankThickScenario",
        args={
            "language": language,
        },
    )

    adapter_spec = get_generation_adapter_spec(
        max_tokens=128,
        stop_sequences=[],
    )

    # Add your annotator spec here - call out annotator for each metric in thick culture -> call cure_annotator.py
    annotator_specs = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.culturalbank_annotator.ThickCultureAgreementAnnotator",
            args={
                "models": ["openai/gpt-4o-mini-2024-07-18"],
                "metric": "agreement",
            }
        ),
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.culturalbank_annotator.ThickCultureCoherenceAnnotator",
            args={
                "models": ["openai/gpt-4o-mini-2024-07-18"],
                "metric": "coherence",
            }
        ),
    ]

    metric_specs = (
        # get_open_ended_generation_metric_specs() # metrics: bleu, rouge, bert_score, 
        get_semantic_similarity_metric_specs() # metrics: cosine similarity
        + get_thick_culture_agreement_exact_match_metric_specs() 
        + get_thick_culture_coherence_metric_specs()
        # + get_toxicity_metric_specs() # metrics: toxicity
    )

    return RunSpec(
        name="culturalbank_thick",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["cure","culturalbank_thick"],
    )

@run_spec_function("casa_thin")
def get_apply_norms_casa_spec(language: str, country: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.cure_scenarios.CASAThinScenario",
        args={
            "language": language,
        },
    )

    adapter_spec = get_generation_adapter_spec(
        max_tokens=5,
        num_outputs=5,
        stop_sequences=[],
    )

    return RunSpec(
        name="casa_thin",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_f1_metric_specs(),
        groups=["cure", "casa_thin"],
    )

@run_spec_function("casa_thick")
def get_reasoning_norms_casa_spec(language: str, country: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.cure_scenarios.CASAThickScenario",
        args={
            "language": language,
        },
    )

    adapter_spec = get_generation_adapter_spec(
        max_tokens=128,
        stop_sequences=[],
    )

    annotator_specs = [
        
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.casa_annotator.ThickCultureAgreementAnnotator",
            args={
                "models": ["openai/gpt-4o-mini-2024-07-18"],
                "metric": "agreement",
            }
        ),
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.casa_annotator.ThickCultureConnotationAnnotator",
            args={
                "models": ["openai/gpt-4o-mini-2024-07-18"],
                "metric": "connotation",
            }
        ),

    ]

    metric_specs = (
        # get_open_ended_generation_metric_specs() # metrics: bleu, rouge, bert_score, 
        get_semantic_similarity_metric_specs() # metrics: cosine similarity
        + get_thick_culture_agreement_exact_match_metric_specs() # Add NLI metric
        + get_thick_culture_connotation_metric_specs()
        # + get_toxicity_metric_specs() # metrics: toxicity
    )

    return RunSpec(
        name="casa_thick",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["cure", "casa_thick"],
    )



@run_spec_function("normad_thin")
def get_normad_thin_spec(language: str, country: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.cure_scenarios.NormADThinScenario",
        args={
            "language": language,
        },
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=INSTRUCTIONS["normad_thin"][language]["instructions"].format(
            country=COUNTRIES[country]
        ),
        input_noun=INSTRUCTIONS["normad_thin"][language]["input_noun"],
        output_noun=INSTRUCTIONS["normad_thin"][language]["output_noun"],
        max_tokens=5,
        num_outputs=5,
        stop_sequences=[],
    )

    return RunSpec(
        name="normad_thin",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs() + get_f1_metric_specs(),
        groups=["cure", "normad_thin"],
    )


@run_spec_function("normad_thick")
def get_normad_thick_spec(language: str, country: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.cure_scenarios.NormADThickScenario",
        args={
            "language": language,
        },
    )
    adapter_spec = get_generation_adapter_spec(
        instructions=INSTRUCTIONS["normad_thick"][language]["instructions"].format(
            country=COUNTRIES[country]
        ),
        input_noun=INSTRUCTIONS["normad_thick"][language]["input_noun"],
        output_noun=INSTRUCTIONS["normad_thick"][language]["output_noun"],
        max_tokens=128,
        stop_sequences=[],
    )
    # Add your annotator spec here - call out annotator for each metric in thick culture -> call cure_annotator.py
    annotator_specs = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.normad_annotator.ThickCultureAgreementAnnotator",
            args={
                "metric": "agreement",
            }
        ),
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.normad_annotator.ThickCultureCoverageAnnotator",
            args={
                "models": ["openai/gpt-4o-mini-2024-07-18"],
                "metric": "coverage",
            }
        ),
    ]

    metric_specs = (
        # get_open_ended_generation_metric_specs() # metrics: bleu, rouge, bert_score, 
        get_semantic_similarity_metric_specs() # metrics: cosine similarity
        + get_thick_culture_agreement_exact_match_metric_specs()
        + get_thick_culture_coverage_metric_specs()
        # + get_toxicity_metric_specs() # metrics: toxicity
    )
    return RunSpec(
        name="normad_thick",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["cure","normad_thick"],
    )


@run_spec_function("nclb_thin")
def get_nclb_thin_spec(language: str, country: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.cure_scenarios.NCLBThinScenario",
        args={
            "language": language,
        },
    )
    adapter_spec = get_generation_adapter_spec(
        max_tokens=5,
        num_outputs=5,
        stop_sequences=[],
    )
    metric_specs = (
        get_exact_match_metric_specs() 
        + get_f1_metric_specs()
    )
    return RunSpec(
        name="nclb_thin",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["cure", "nclb_thin"],
    )





@run_spec_function("nclb_thick")
def get_nclb_thick_spec(language: str, country: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.cure_scenarios.NCLBThickScenario",
        args={
            "language": language,
        },
    )
    adapter_spec = get_generation_adapter_spec(
        max_tokens=128,
        stop_sequences=[],
    )
    # Add your annotator spec here - call out annotator for each metric in thick culture -> call cure_annotator.py
    annotator_specs = [
        AnnotatorSpec(
            class_name="helm.benchmark.annotation.nclb_annotator.ThickCultureAgreementAnnotator",
            args={
                "metric": "agreement",
            }
        ),
    ]

    metric_specs = (
        # get_open_ended_generation_metric_specs() # metrics: bleu, rouge, bert_score, 
        get_semantic_similarity_metric_specs() # metrics: cosine similarity
        + get_thick_culture_agreement_exact_match_metric_specs()
        + get_thick_culture_specificity_metric_specs()
        # + get_toxicity_metric_specs() # metrics: toxicity
    )
    return RunSpec(
        name="nclb_thick",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        annotators=annotator_specs,
        metric_specs=metric_specs,
        groups=["cure","nclb_thick"],
    )













# @run_spec_function("daily_dilemmas_thin")
# def get_daily_dilemmas_thin_spec(language: str, country: str) -> RunSpec:
#     scenario_spec = ScenarioSpec(
#         class_name="helm.benchmark.scenarios.cure_scenarios.DailyDilemmasThinScenario",
#         args={
#             "language": language,
#         },
#     )

#     adapter_spec = get_generation_adapter_spec(
#         max_tokens=5,
#         num_outputs=5,
#         stop_sequences=[],
#     )

#     return RunSpec(
#         name="daily_dilemmas_thin",
#         scenario_spec=scenario_spec,
#         adapter_spec=adapter_spec,
#         metric_specs=get_exact_match_metric_specs() + get_f1_metric_specs(),
#         groups=["cure", "daily_dilemmas_thin"],
#     )



# @run_spec_function("daily_dilemmas_thick")
# def get_daily_dilemmas_thick_spec(language: str, country: str) -> RunSpec:
#     scenario_spec = ScenarioSpec(
#         class_name="helm.benchmark.scenarios.cure_scenarios.DailyDilemmasThickScenario",
#         args={
#             "language": language,
#         },
#     )

#     adapter_spec = get_generation_adapter_spec(
#         max_tokens=128,
#         stop_sequences=[],
#     )

#     # Add your annotator spec here - call out annotator for each metric in thick culture -> call cure_annotator.py
#     annotator_specs = [
#         AnnotatorSpec(
#             class_name="helm.benchmark.annotation.normad_annotator.ThickCultureCoverageAnnotator",
#             args={
#                 "models": ["openai/gpt-4o-mini-2024-07-18"],
#                 "metric": "coverage",
#             }
#         ),
#     ]

#     metric_specs = (
#         get_open_ended_generation_metric_specs() # metrics: bleu, rouge, bert_score, 
#         + get_semantic_similarity_metric_specs() # metrics: cosine similarity
#         + get_thick_culture_coverage_metric_specs()
#         + get_toxicity_metric_specs() # metrics: toxicity
#     )

#     return RunSpec(
#         name="daily_dilemmas_thick",
#         scenario_spec=scenario_spec,
#         adapter_spec=adapter_spec,
#         annotators=annotator_specs,
#         metric_specs=metric_specs,
#         groups=["cure", "daily_dilemmas_thick"],
#     )
#         + get_toxicity_metric_specs() # metrics: toxicity
#     )

#     return RunSpec(
#         name="daily_dilemmas_thick",
#         scenario_spec=scenario_spec,
#         adapter_spec=adapter_spec,
#         annotators=annotator_specs,
#         metric_specs=metric_specs,
#         groups=["cure", "daily_dilemmas_thick"],
#     )

