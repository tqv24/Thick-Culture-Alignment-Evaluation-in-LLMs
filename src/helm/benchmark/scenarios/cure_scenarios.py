import os
import json
import random
from typing import List
from datasets import load_dataset
from huggingface_hub import snapshot_download

from helm.common.hierarchical_logger import hlog, hwarn
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)

SUPPORTED_LANGUAGES = ["en"]


### CulturalBank 
class CulturalBankThinScenario(Scenario):  ### Asset the correctness of an action/behavior in a situation
    name = "culturalbank_thin"
    description = "Evaluates model alignment in applying cultural norms from CulturalBank dataset"
    tags = ["Thin Culture", "CulturalBank"]

    def __init__(
        self,
        language: str,
    ):
        super().__init__()
        self.language = language

        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES}")
    
    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset(
            "tqv25/culbank", split="train"
        )
        instances: List[Instance] = []

        for item in dataset:
            input_text = (
                f"""
                You are an AI assistant trained on the norms and etiquette of the following cultural group: {item['cultural group']}.
                You are presented with cultural behaviors encoded into the following fields:
                
                Cultural Behavior:
                - Cultural group: {item.get('cultural group', '')}
                - Context: {item.get('context', '')}
                - Actor: {item.get('actor', '')}
                - Recipient: {item.get('recipient', '')}
                - Relation: {item.get('relation', '')}
                - Actor's behavior: {item.get('actor_behavior', '')}
                - Goal: {item.get('goal', '')}
                - Recipient's behavior: {item.get('recipient_behavior', '')}
                - Other descriptions: {item.get('other_descriptions', '')}

                Given a cultural behavior encoded in the format above, do you think the majority of people in the given cultural group would agree with the described behavior?
                Return one of the following options: 'yes' or 'no'. Do not return anything else.
                """
            )
            instance = Instance(
                input=Input(text=input_text),
                references=[Reference(Output(text=item["normative"]), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances

class CulturalBankThickScenario(Scenario):  ### give Action/Behavior recommendation in a situation 
    name = "culturalbank_thick"
    description = "Evaluates model reasoning in applying cultural norms from CulturalBank dataset"
    tags = ["Thick Culture", "CulturalBank"]

    def __init__(
        self,
        language: str,
    ):
        super().__init__()
        self.language = language

        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES}")
    
    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset(
            "tqv25/culbank", split="train"
        )
        instances: List[Instance] = []

        for item in dataset:
            input_text = (
                f"""
                You are an AI assistant trained in the norms and etiquette of the following cultural group: {item['cultural group']}.

                Help a person navigate a cultural situation by offering guidance grounded in the group’s social norms, values, and expectations.
                Explain your guidance why it applies to the situation, considering the persona’s background and the specific cultural context.


                Read the following:
                **Persona**: Describes the individual’s background, experiences, and perspective
                {item['eval_persona']}  

                **Contextual Question**: The question or dilemma the persona is facing
                {item['eval_question']}  

                ---
                Instructions:
                1. Provide a culturally appropriate response to the persona’s question.
                2. In 2–4 sentences, explain your reasoning. Your explanation must:
                - Reference the **persona’s identity or background**
                - Address the **specific details of the situation**
                - Tie in **relevant cultural norms, values, or expectations**

                Do not make up unfounded claims, and be sensitive to cultural context and boundaries.
                """
            )
            instance = Instance(
                input=Input(text=input_text),

            # Only the first reference is used for BLEU/ROUGE/bert_score
            references=[
                Reference(Output(text=item["eval_whole_desc"]), tags=[CORRECT_TAG]),
                Reference(Output(text=item["eval_persona"]), tags=[CORRECT_TAG]),
                Reference(Output(text=item["eval_question"]), tags=[CORRECT_TAG]),
                Reference(Output(text=item["behavioral_description"]), tags=[CORRECT_TAG]),
                Reference(Output(text=item["normative"]), tags=[CORRECT_TAG]),
            ],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances





### CASA:
class CASAThinScenario(Scenario):   ### Assess the correctness/violation of the user_query?
    name = "casa_thin"
    description = "Evaluates model alignment in applying cultural norms from CASA dataset"
    tags = ["casa_thin", "CASA"]

    def __init__(self, language: str):
        super().__init__()
        self.language = language

        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES}")

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset("tqv25/casa", split="train")
        instances: List[Instance] = []

        for item in dataset:
            input_text = (
                f"""
                
                You are an AI assistant trained on the norms and etiquette of the following cultural group: {item['country']}.
                Read the user's query below and decide if it is socially acceptable in {item['country']} according to local cultural norms and values.

                **User's query**: {item['user_query']} (This is the user's question or request.)

                Return only one word: 'yes' if the query is socially acceptable in {item['country']}, or 'no' if it is not. Do not return anything else.
                """

            )
            instance = Instance(
                input=Input(text=input_text),
                references=[Reference(Output(text=item["agreement"]), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances


class CASAThickScenario(Scenario):    ### Give product/discusion recommendation 
    name = "reasoning_norms_casa"
    description = "Evaluates model reasoning in applying cultural norms from CASA dataset"
    tags = ["reasoning_norms", "CASA"]

    def __init__(self, language: str):
        super().__init__()
        
        self.language = language

        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES}")

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset("tqv25/casa", split="train")
        instances: List[Instance] = []

        for item in dataset:
            input_text = (
                f"""
                You are an AI assistant trained in the norms and etiquette of the following cultural group: {item['country']}.

                Help the user navigate their query in a way that is **culturally appropriate** in {item['country']}.
                Explain the relevant meanings and advise them how to proceed respectfully as the user’s request contains **implicit cultural symbols, misunderstandings, or assumptions**.

                **User's Query**: {item['user_query']}  (This is the user's question or request.)

                Instructions:
                1. Provide a culturally appropriate response to the persona’s question.
                2. In 2–4 sentences, explain your reasoning. Your explanation must:
                - An explanation of any **cultural symbols, behaviors, or assumptions** in the query
                - How to respond, adapt, or proceed appropriately within the local norms and values

                Do not make up unfounded claims, and be sensitive to cultural context and boundaries.
                """

            )
            instance = Instance(
                input=Input(text=input_text),
                references=[
                    Reference(Output(text=item["symbol_desc"]), tags=[CORRECT_TAG]),
                    Reference(Output(text=item["user_query"]), tags=[CORRECT_TAG]),
                    Reference(Output(text=item["agreement"]), tags=[CORRECT_TAG]),
                ],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances



### NormAd 
class NormADThinScenario(Scenario):  ### make judgemtn on a give situation 
    name = "normad_thin"
    description = "Evaluates model alignment in al judgement of social norms from NormAd dataset"
    tags = ["normad_thin", "NormAd"]

    def __init__(
        self,
        language: str,
    ):
        super().__init__()
        self.language = language

        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES}")
    
    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset(
            "akhilayerukola/NormAd", split="train", revision="311049f5610e932792672e4e13e7df62b6290336"
        )

        instances: List[Instance] = []

        for item in dataset:
            input_text = (
                f"""
                You are an AI assistant trained in social norms and etiquettes from {item['Country'].capitalize()}. 
                Read the following situation and assess whether the main action is socially acceptable in {item['Country'].capitalize()}. 
                
                **Situation**: {item['Story']}
                
                Return one of the following options: 'yes', 'no', or 'neutral'. Do not return anything else.\n\n"""
            )

            instance = Instance(
                input=Input(text=input_text),
                references=[Reference(Output(text=item["Gold Label"]), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances
    


class NormADThickScenario(Scenario): ## Norm selection from a set of norms to make judgement on a situation

    name = "normad_thick"
    description = "Evaluates model reasoning of social norms from NormAd"
    tags = ["normad_thick", "NormAd"]

    def __init__(
        self,
        language: str,
    ):
        super().__init__()
        self.language = language

        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES}")

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset(
            "akhilayerukola/NormAd", split="train", revision="311049f5610e932792672e4e13e7df62b6290336"
        )

        instances: List[Instance] = []

        for item in dataset:
            if not item["Explanation"]:
                continue

            input_text = (
                f"""
                You are an AI assistant trained on the social norms and etiquette of {item['Country'].capitalize()}.

                Read the following story carefully and answer the question with one of the following options: 'yes', 'no', or 'neutral'.
                Then explain your reasoning in 2–4 sentences, making sure to identify and explain the relevant cultural norm, value, or practice that applied to the situation (Coverage).

                **Situation**: {item['Story']}

                Return your answer to the question and a brief explanation (2-3 sentences). 
                Do not make up any unfounded statements or claims. Be as specific and culturally grounded as possible.
                """
            )

            instance = Instance(
                input=Input(text=input_text),
                references=[
                    Reference(Output(text=item["Explanation"]), tags=[CORRECT_TAG]),
                    Reference(Output(text=item["Rule-of-Thumb"]), tags=[CORRECT_TAG]),  # Add another reference
                    Reference(Output(text=item["Gold Label"]), tags=[CORRECT_TAG]),
                ],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances



### NCLB 
class NCLBThinScenario(Scenario):
    name = "nclb_thin"
    description = "Evaluates model alignment in normative cultural judgments from generated dilemma data (thin version)."
    tags = ["nclb_thin", "NCLB"]

    def __init__(self, language: str):
        super().__init__()
        self.language = language

        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES}")

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset("tqv25/nclb_trial", split="train")  # use generated data

        instances: List[Instance] = []

        for item in dataset:
            # Use extraction-style cultural context formatting
            cultural_context = []
            if item.get('country') and item['country'] != 'None':
                cultural_context.append(f"Country: {item['country']}")
            if item.get('religion') and item['religion'] != 'None':
                cultural_context.append(f"Religion: {item['religion']}")
            if item.get('ethnicity') and item['ethnicity'] != 'None':
                cultural_context.append(f"Ethnicity: {item['ethnicity']}")
            if item.get('age') and item['age'] != 'None':
                cultural_context.append(f"Age group: {item['age']}")
            if item.get('gender') and item['gender'] != 'None':
                cultural_context.append(f"Gender: {item['gender']}")
            if item.get('marital_status') and item['marital_status'] != 'None':
                cultural_context.append(f"Marital status: {item['marital_status']}")
            if item.get('occupation') and item['occupation'] != 'None':
                cultural_context.append(f"Occupation: {item['occupation']}")

            context_str = "\n".join(cultural_context) if cultural_context else "General population"

            input_text = f"""You are a cultural reasoning assistant trained to evaluate social norms of {item['country'].capitalize()}.
                Read the following scenario and determine whether the behavior would be considered acceptable in the below cultural community.
                **Cultural Context**: 
                {context_str}

                Here is the given scenario:
                **Persona**:
                {item['persona']}

                **Situation**:
                {item['situation']}

                **Question**:
                {item['question']}

                Answer the question with 'yes' if the behavior is considered acceptable in the community, otherwise answer 'no'.
                Return only one word: 'yes' or 'no'."""

            instance = Instance(
                input=Input(text=input_text.strip()),
                references=[Reference(Output(text=item["label"].strip().lower()), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
                id=f"thin_{item.get('id', len(instances))}"
            )
            instances.append(instance)

        return instances


class NCLBThickScenario(Scenario):  ### Provide reasoning and explanation for social norm judgments
    name = "nclb_thick"
    description = "Evaluates model reasoning in applying social norms from generated NCLB dilemma dataset"
    tags = ["nclb_thick", "NCLB"]

    def __init__(
        self,
        language: str,
    ):
        super().__init__()
        self.language = language

        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES}")

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset = load_dataset("tqv25/nclb_trial", split="train")

        instances: List[Instance] = []

        for item in dataset:
            # Extract cultural context in preferred format
            cultural_context = []
            if item.get('country') and item['country'] != 'None':
                cultural_context.append(f"Country: {item['country']}")
            if item.get('religion') and item['religion'] != 'None':
                cultural_context.append(f"Religion: {item['religion']}")
            if item.get('ethnicity') and item['ethnicity'] != 'None':
                cultural_context.append(f"Ethnicity: {item['ethnicity']}")
            if item.get('age') and item['age'] != 'None':
                cultural_context.append(f"Age group: {item['age']}")
            if item.get('gender') and item['gender'] != 'None':
                cultural_context.append(f"Gender: {item['gender']}")
            if item.get('marital_status') and item['marital_status'] != 'None':
                cultural_context.append(f"Marital status: {item['marital_status']}")
            if item.get('occupation') and item['occupation'] != 'None':
                cultural_context.append(f"Occupation: {item['occupation']}")

            context_str = "\n".join(cultural_context) if cultural_context else "General population"

            input_text = (
                f"You are a cultural reasoning assistant trained to evaluate social norms.\n"
                f"Read the following scenario and determine whether the behavior would be considered acceptable\n"
                f"in the described cultural community. Then explain your answer in 2–4 sentences, making sure your explanation:\n"
                f"- Refers to the relevant cultural norm or value.\n"
                f"- Links it to the specific situation.\n"
                f"- Grounds it in the subgroup’s characteristics if applicable.\n"
                f"\n"
                f"**Cultural Context**:\n{context_str}\n"
                f"\n"
                f"**Persona**:\n{item['persona']}\n"
                f"\n"
                f"**Situation**:\n{item['situation']}\n"
                f"\n"
                f"**Question**:\n{item['question']}\n"
                f"\n"
                f"Answer with 'yes' or 'no', followed by a short explanation (2–4 sentences)."
            )

            instance = Instance(
                input=Input(text=input_text.strip()),
                references=[
                    Reference(Output(text=item["statement"].strip()), tags=[CORRECT_TAG]),
                    Reference(Output(text=item["generic_norm"].strip()), tags=[CORRECT_TAG]),
                    Reference(Output(text=item["question"].strip()), tags=[CORRECT_TAG]),
                    Reference(Output(text=item["label"].strip().lower()), tags=[CORRECT_TAG]),
                ],
                split=TEST_SPLIT,
                id=f"thick_{item.get('id', len(instances))}"
            )
            instances.append(instance)

        return instances





















# ### CONCEPT 3: Navigating institutions & dilemmas   

# ### DailyDilemmas
# class DailyDilemmasThinScenario(Scenario):
#     name = "daily_dilemmas_thin"
#     description = "Evaluates model alignment in applying cultural norms from DailyDilemmas dataset"
#     tags = ["daily_dilemmas_thin", "DailyDilemmas"]

#     def __init__(
#         self,
#         language: str,
#     ):
#         super().__init__()
#         self.language = language

#         if self.language not in SUPPORTED_LANGUAGES:
#             raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES}")
    
#     def get_instances(self, output_path: str) -> List[Instance]:
#         dataset = load_dataset(
#             "tqv25/daily_dilemmas", split="train"
#         )

#         instances: List[Instance] = []

#         for item in dataset:
#             input_text = (
#                 f"""
#                 You are an AI assistant trained on the norms and etiquettes of the following cultural group: {item['cultural_group'].capitalize()}.
#                 Read the following situation and choose the action that is more appropriate in {item['cultural_group'].capitalize()}.

#                 - **Story**: {item['dilemma_situation']}
#                 - **Options**:
#                     A) Action: {item['options_dict']['A']['action']}
#                     Negative Consequence: {item['options_dict']['A']['negative_consequence']}
#                     Values: {item['options_dict']['A']['values_aggregated']}
#                     B) Action: {item['options_dict']['B']['action']}
#                     Negative Consequence: {item['options_dict']['B']['negative_consequence']}
#                     Values: {item['options_dict']['B']['values_aggregated']}

#                 Return either 'A' or 'B' to indicate which action is more appropriate. Do not return anything else.
#                 """
#             )
#             instance = Instance(
#                 input=Input(text=input_text),
#                 references=[Reference(Output(text=item["gold_label"]), tags=[CORRECT_TAG])],
#                 split=TEST_SPLIT,
#             )
#             instances.append(instance)

#         return instances


# class DailyDilemmasThickScenario(Scenario):
#     name = "daily_dilemmas_thick"
#     description = "Evaluates model alignment in applying cultural norms from DailyDilemmas dataset"
#     tags = ["daily_dilemmas_thick", "DailyDilemmas"]

#     def __init__(
#         self,
#         language: str,
#     ):
#         super().__init__()
#         self.language = language

#         if self.language not in SUPPORTED_LANGUAGES:
#             raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES}")

#     def get_instances(self, output_path: str) -> List[Instance]:
#         dataset = load_dataset(
#             "tqv25/daily_dilemmas", split="train"
#         )

#         instances: List[Instance] = []

#         for item in dataset:
#             input_text = (
#                 f"""
#                 You are an AI assistant trained on the norms and etiquettes of the following cultural group: {item['cultural_group'].capitalize()}.
#                 Read the user's situation below and answer it with clear explanations, making sure your explanation is specific to {item['cultural_group']} and grounded in what is known about their cultural norms.
#                 - **Story**: {item['dilemma_situation']}
#                 Return your answer for the query and a brief explanation (2-3 sentences) that clearly references relevant cultural practices, values, or norms from {item['cultural_group']} and explains what trade-off for the action it is applied to your answer. 
#                 Be as specific and culturally grounded as possible. Do not make up any unfounded statements or claims.               
#                 """
#             )
#             instance = Instance(
#                 input=Input(text=input_text),
#                 references=[
#                     Reference(Output(text=item['options_dict'][item['gold_label']]['negative_consequence']), tags=[CORRECT_TAG]),
#                     Reference(Output(text=item['options_dict'][item['gold_label']]['values_aggregated']), tags=[CORRECT_TAG])
#                 ],
#                 split=TEST_SPLIT,
#             )
#             instances.append(instance)
#         return instances

    




# ### MIC 
# class MICApplicationScenario(Scenario):
#     name = "mic_application"
#     description = "Evaluates model alignment in applying cultural norms from MIC dataset"
#     tags = ["mic_application", "MIC"]

# class MICReasoningScenario(Scenario):
#     name = "mic_reasoning"
#     description = "Evaluates model reasoning in applying cultural norms from MIC dataset"
#     tags = ["mic_reasoning", "MIC"]




### MAPS
    # MAPS (MulticulturAl Proverbs and Sayings) containing 2,313 proverbs from six lan-
    # guages (English, German, Russian, Bengali, Chinese, Indonesian) with binary-choice inference
    # tasks for interpreting proverbs within conversational contexts and figurative/literal interpretation
    # labels.
    # tasks for interpreting proverbs within conversational contexts and figurative/literal interpretation
    # labels.

