#!/usr/bin/env python3
"""Comprehensive evaluation system for MOSDAC chatbot.

Evaluates intent recognition, entity recognition, response completeness,
response consistency, and overall system performance.
"""

import json
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from rouge_score import rouge_scorer
from sacrebleu import BLEU
import matplotlib.pyplot as plt
import seaborn as sns
from seqeval.metrics import classification_report as seq_classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    intent_accuracy: float
    entity_f1: float
    response_completeness: float
    response_consistency: float
    retrieval_precision: float
    retrieval_recall: float
    generation_bleu: float
    generation_rouge_l: float
    overall_score: float


class IntentEvaluator:
    """Evaluate intent recognition accuracy."""
    
    def __init__(self):
        self.test_cases = [
            ("What satellite data is available?", "data_availability"),
            ("How do I download INSAT images?", "download_help"),
            ("What is the resolution of OCEANSAT?", "product_specs"),
            ("Show me weather data", "data_request"),
            ("I need technical support", "support"),
            ("Product information for SCATSAT", "product_info"),
            ("How to register account?", "registration"),
            ("Coverage area information", "coverage_info"),
            ("Available data formats", "documentation"),
            ("Help with data access", "support")
        ]
    
    def evaluate(self, predict_intent_fn) -> Dict[str, float]:
        """Evaluate intent classification performance."""
        y_true = []
        y_pred = []
        
        for text, true_intent in self.test_cases:
            predicted_intent = predict_intent_fn(text)
            y_true.append(true_intent)
            y_pred.append(predicted_intent)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        return {
            "intent_accuracy": accuracy,
            "intent_precision": precision,
            "intent_recall": recall,
            "intent_f1": f1
        }


class EntityEvaluator:
    """Evaluate named entity recognition performance."""
    
    def __init__(self):
        self.test_cases = [
            ("Show me INSAT-3D temperature data", [("INSAT-3D", "SATELLITE"), ("temperature", "PARAMETER")]),
            ("OCEANSAT-2 ocean color products", [("OCEANSAT-2", "SATELLITE"), ("ocean color", "PARAMETER")]),
            ("Data for Mumbai region", [("Mumbai", "LOCATION")]),
            ("SCATSAT-1 wind measurements", [("SCATSAT-1", "SATELLITE"), ("wind", "PARAMETER")]),
            ("Bay of Bengal cyclone data", [("Bay of Bengal", "LOCATION"), ("cyclone", "PARAMETER")])
        ]
    
    def evaluate(self, extract_entities_fn) -> Dict[str, float]:
        """Evaluate NER performance."""
        y_true = []
        y_pred = []
        
        for text, true_entities in self.test_cases:
            predicted_entities = extract_entities_fn(text)
            
            # Convert to BIO format for evaluation
            true_labels = self._entities_to_bio(text, true_entities)
            pred_labels = self._entities_to_bio(text, predicted_entities)
            
            y_true.append(true_labels)
            y_pred.append(pred_labels)
        
        # Use seqeval for sequence labeling metrics
        report = seq_classification_report(y_true, y_pred, output_dict=True)
        
        return {
            "entity_precision": report['micro avg']['precision'],
            "entity_recall": report['micro avg']['recall'],
            "entity_f1": report['micro avg']['f1-score']
        }
    
    def _entities_to_bio(self, text: str, entities: List[Tuple[str, str]]) -> List[str]:
        """Convert entities to BIO tagging format."""
        tokens = text.split()
        labels = ['O'] * len(tokens)
        
        for entity_text, entity_type in entities:
            # Simple token matching (would need better alignment in practice)
            for i, token in enumerate(tokens):
                if entity_text.lower() in token.lower():
                    labels[i] = f'B-{entity_type}'
                    
        return labels


class ResponseCompletenessEvaluator:
    """Evaluate how complete responses are relative to query context."""
    
    def __init__(self):
        self.test_cases = [
            {
                "question": "What is INSAT-3D and what data does it provide?",
                "expected_aspects": ["satellite_type", "data_products", "coverage", "resolution"],
                "reference_answer": "INSAT-3D is a geostationary meteorological satellite that provides temperature, humidity, and cloud imagery data with 4km resolution covering the Indian Ocean region."
            },
            {
                "question": "How to download OCEANSAT-2 ocean color data?",
                "expected_aspects": ["registration", "data_format", "download_process", "access_portal"],
                "reference_answer": "Register on MOSDAC portal, navigate to OCEANSAT-2 section, select ocean color products, choose format and download via FTP or HTTP."
            }
        ]
    
    def evaluate(self, generate_response_fn) -> Dict[str, float]:
        """Evaluate response completeness."""
        completeness_scores = []
        
        for test_case in self.test_cases:
            question = test_case["question"]
            expected_aspects = test_case["expected_aspects"]
            
            response = generate_response_fn(question)
            
            # Simple keyword-based completeness check
            covered_aspects = 0
            for aspect in expected_aspects:
                if any(keyword in response.lower() for keyword in aspect.split('_')):
                    covered_aspects += 1
            
            completeness = covered_aspects / len(expected_aspects)
            completeness_scores.append(completeness)
        
        return {
            "response_completeness": np.mean(completeness_scores),
            "completeness_std": np.std(completeness_scores)
        }


class ResponseConsistencyEvaluator:
    """Evaluate consistency across multi-turn conversations."""
    
    def __init__(self):
        self.conversation_scenarios = [
            {
                "turns": [
                    "What is INSAT-3D?",
                    "What data products does it provide?",
                    "What is its spatial resolution?"
                ],
                "consistency_checks": [
                    ("satellite_name", "INSAT-3D"),
                    ("data_type", "meteorological")
                ]
            }
        ]
    
    def evaluate(self, chat_session_fn) -> Dict[str, float]:
        """Evaluate multi-turn consistency."""
        consistency_scores = []
        
        for scenario in self.conversation_scenarios:
            responses = []
            for turn in scenario["turns"]:
                response = chat_session_fn(turn)
                responses.append(response)
            
            # Check consistency across responses
            consistency_score = self._check_consistency(responses, scenario["consistency_checks"])
            consistency_scores.append(consistency_score)
        
        return {
            "response_consistency": np.mean(consistency_scores),
            "consistency_std": np.std(consistency_scores)
        }
    
    def _check_consistency(self, responses: List[str], checks: List[Tuple[str, str]]) -> float:
        """Check consistency of information across responses."""
        consistent_checks = 0
        
        for check_type, expected_value in checks:
            mentions = [expected_value.lower() in response.lower() for response in responses]
            if all(mentions) or not any(mentions):  # All mention or none mention
                consistent_checks += 1
        
        return consistent_checks / len(checks) if checks else 1.0


class RetrievalEvaluator:
    """Evaluate retrieval component performance."""
    
    def __init__(self):
        self.test_queries = [
            {
                "query": "INSAT-3D temperature data",
                "relevant_docs": ["doc_1", "doc_3", "doc_7"],
                "total_docs": 10
            },
            {
                "query": "OCEANSAT-2 ocean color",
                "relevant_docs": ["doc_2", "doc_5"],
                "total_docs": 10
            }
        ]
    
    def evaluate(self, retrieve_docs_fn) -> Dict[str, float]:
        """Evaluate retrieval metrics."""
        precisions = []
        recalls = []
        
        for test_case in self.test_queries:
            query = test_case["query"]
            relevant_docs = set(test_case["relevant_docs"])
            
            retrieved_docs = set(retrieve_docs_fn(query, k=5))
            
            if retrieved_docs:
                precision = len(relevant_docs & retrieved_docs) / len(retrieved_docs)
                recall = len(relevant_docs & retrieved_docs) / len(relevant_docs)
            else:
                precision = recall = 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        return {
            "retrieval_precision": np.mean(precisions),
            "retrieval_recall": np.mean(recalls),
            "retrieval_f1": 2 * np.mean(precisions) * np.mean(recalls) / (np.mean(precisions) + np.mean(recalls)) if (np.mean(precisions) + np.mean(recalls)) > 0 else 0
        }


class GenerationEvaluator:
    """Evaluate response generation quality."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU()
        
        self.test_cases = [
            {
                "question": "What is INSAT-3D?",
                "reference": "INSAT-3D is an Indian geostationary meteorological satellite that provides weather and climate data including temperature, humidity, and cloud imagery with 4km spatial resolution."
            },
            {
                "question": "How to download satellite data?",
                "reference": "To download satellite data from MOSDAC, first register on the portal, then navigate to the data section, select your desired products, choose format, and download via FTP or HTTP."
            }
        ]
    
    def evaluate(self, generate_response_fn) -> Dict[str, float]:
        """Evaluate generation quality using ROUGE and BLEU."""
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        bleu_scores = []
        
        for test_case in self.test_cases:
            question = test_case["question"]
            reference = test_case["reference"]
            
            generated = generate_response_fn(question)
            
            # ROUGE scores
            rouge_result = self.rouge_scorer.score(reference, generated)
            for metric in rouge_scores:
                rouge_scores[metric].append(rouge_result[metric].fmeasure)
            
            # BLEU score
            bleu_score = self.bleu.sentence_score(generated, [reference]).score / 100.0
            bleu_scores.append(bleu_score)
        
        return {
            "generation_rouge1": np.mean(rouge_scores['rouge1']),
            "generation_rouge2": np.mean(rouge_scores['rouge2']),
            "generation_rougeL": np.mean(rouge_scores['rougeL']),
            "generation_bleu": np.mean(bleu_scores)
        }


class SystemEvaluator:
    """Comprehensive system evaluation."""
    
    def __init__(self):
        self.intent_evaluator = IntentEvaluator()
        self.entity_evaluator = EntityEvaluator()
        self.completeness_evaluator = ResponseCompletenessEvaluator()
        self.consistency_evaluator = ResponseConsistencyEvaluator()
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
    
    def evaluate_full_system(self, system_components: Dict) -> EvaluationResults:
        """Run comprehensive evaluation of the entire system."""
        logger.info("Starting comprehensive system evaluation...")
        
        # Extract component functions
        predict_intent = system_components.get('predict_intent')
        extract_entities = system_components.get('extract_entities')
        generate_response = system_components.get('generate_response')
        chat_session = system_components.get('chat_session')
        retrieve_docs = system_components.get('retrieve_docs')
        
        results = {}
        
        # Intent evaluation
        if predict_intent:
            intent_results = self.intent_evaluator.evaluate(predict_intent)
            results.update(intent_results)
        
        # Entity evaluation
        if extract_entities:
            entity_results = self.entity_evaluator.evaluate(extract_entities)
            results.update(entity_results)
        
        # Response completeness
        if generate_response:
            completeness_results = self.completeness_evaluator.evaluate(generate_response)
            results.update(completeness_results)
        
        # Response consistency
        if chat_session:
            consistency_results = self.consistency_evaluator.evaluate(chat_session)
            results.update(consistency_results)
        
        # Retrieval evaluation
        if retrieve_docs:
            retrieval_results = self.retrieval_evaluator.evaluate(retrieve_docs)
            results.update(retrieval_results)
        
        # Generation evaluation
        if generate_response:
            generation_results = self.generation_evaluator.evaluate(generate_response)
            results.update(generation_results)
        
        # Calculate overall score
        key_metrics = [
            results.get('intent_accuracy', 0),
            results.get('entity_f1', 0),
            results.get('response_completeness', 0),
            results.get('response_consistency', 0),
            results.get('retrieval_f1', 0),
            results.get('generation_rougeL', 0)
        ]
        overall_score = np.mean([m for m in key_metrics if m > 0])
        results['overall_score'] = overall_score
        
        # Create results object
        evaluation_results = EvaluationResults(
            intent_accuracy=results.get('intent_accuracy', 0),
            entity_f1=results.get('entity_f1', 0),
            response_completeness=results.get('response_completeness', 0),
            response_consistency=results.get('response_consistency', 0),
            retrieval_precision=results.get('retrieval_precision', 0),
            retrieval_recall=results.get('retrieval_recall', 0),
            generation_bleu=results.get('generation_bleu', 0),
            generation_rouge_l=results.get('generation_rougeL', 0),
            overall_score=overall_score
        )
        
        # Save detailed results
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualization
        self._create_evaluation_report(results)
        
        logger.info(f"Evaluation completed. Overall score: {overall_score:.3f}")
        return evaluation_results
    
    def _create_evaluation_report(self, results: Dict[str, float]):
        """Create visual evaluation report."""
        # Create metrics plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Intent metrics
        intent_metrics = {k: v for k, v in results.items() if 'intent' in k}
        if intent_metrics:
            axes[0, 0].bar(intent_metrics.keys(), intent_metrics.values())
            axes[0, 0].set_title('Intent Recognition Metrics')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Entity metrics
        entity_metrics = {k: v for k, v in results.items() if 'entity' in k}
        if entity_metrics:
            axes[0, 1].bar(entity_metrics.keys(), entity_metrics.values())
            axes[0, 1].set_title('Entity Recognition Metrics')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Retrieval metrics
        retrieval_metrics = {k: v for k, v in results.items() if 'retrieval' in k}
        if retrieval_metrics:
            axes[1, 0].bar(retrieval_metrics.keys(), retrieval_metrics.values())
            axes[1, 0].set_title('Retrieval Metrics')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Generation metrics
        generation_metrics = {k: v for k, v in results.items() if 'generation' in k}
        if generation_metrics:
            axes[1, 1].bar(generation_metrics.keys(), generation_metrics.values())
            axes[1, 1].set_title('Generation Metrics')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('evaluation_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Evaluation report saved as 'evaluation_report.png'")


# Example usage
if __name__ == "__main__":
    # Mock system components for demonstration
    def mock_predict_intent(text):
        if "download" in text.lower():
            return "download_help"
        elif "data" in text.lower():
            return "data_availability"
        else:
            return "support"
    
    def mock_extract_entities(text):
        entities = []
        if "INSAT" in text:
            entities.append(("INSAT-3D", "SATELLITE"))
        if "temperature" in text:
            entities.append(("temperature", "PARAMETER"))
        return entities
    
    def mock_generate_response(question):
        return f"This is a response to: {question}"
    
    def mock_chat_session(turn):
        return f"Session response to: {turn}"
    
    def mock_retrieve_docs(query, k=5):
        return [f"doc_{i}" for i in range(k)]
    
    # Run evaluation
    system_components = {
        'predict_intent': mock_predict_intent,
        'extract_entities': mock_extract_entities,
        'generate_response': mock_generate_response,
        'chat_session': mock_chat_session,
        'retrieve_docs': mock_retrieve_docs
    }
    
    evaluator = SystemEvaluator()
    results = evaluator.evaluate_full_system(system_components)
    
    print(f"Evaluation Results:")
    print(f"Overall Score: {results.overall_score:.3f}")
    print(f"Intent Accuracy: {results.intent_accuracy:.3f}")
    print(f"Entity F1: {results.entity_f1:.3f}")
    print(f"Response Completeness: {results.response_completeness:.3f}")
    print(f"Response Consistency: {results.response_consistency:.3f}")