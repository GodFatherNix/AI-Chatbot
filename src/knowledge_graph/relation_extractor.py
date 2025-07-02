"""
Relation extraction for MOSDAC knowledge graph construction.
"""
import re
import spacy
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import json
from datetime import datetime
import hashlib
from itertools import combinations

from .entity_extractor import Entity
from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

@dataclass
class Relation:
    """Represents a relationship between entities."""
    id: str
    subject_entity: Entity
    predicate: str
    object_entity: Entity
    confidence: float
    context: str
    source_sentence: str
    attributes: Dict[str, Any]

class RelationExtractor:
    """
    Extracts relationships between entities for knowledge graph construction.
    Uses pattern-based rules and dependency parsing.
    """
    
    def __init__(self):
        self.nlp = None
        self._load_spacy_model()
        
        # Relation types from configuration
        self.relation_types = config.get('knowledge_graph.relations', [
            'is_part_of', 'provides', 'located_at', 'operates_on', 
            'related_to', 'supersedes'
        ])
        
        # Define relation patterns for different entity type combinations
        self.relation_patterns = {
            ('Satellite', 'Organization'): [
                {
                    'pattern': r'{subject}.*(?:operated|launched|developed|built).*by.*{object}',
                    'relation': 'operated_by'
                },
                {
                    'pattern': r'{object}.*(?:operates|launched|developed|built).*{subject}',
                    'relation': 'operates'
                }
            ],
            ('Satellite', 'Technology'): [
                {
                    'pattern': r'{subject}.*(?:carries|equipped with|uses|has).*{object}',
                    'relation': 'carries'
                },
                {
                    'pattern': r'{object}.*(?:onboard|on).*{subject}',
                    'relation': 'installed_on'
                }
            ],
            ('Satellite', 'Mission'): [
                {
                    'pattern': r'{subject}.*(?:part of|supports|used in).*{object}',
                    'relation': 'supports'
                },
                {
                    'pattern': r'{object}.*(?:uses|utilizes).*{subject}',
                    'relation': 'utilizes'
                }
            ],
            ('Technology', 'Product'): [
                {
                    'pattern': r'{subject}.*(?:produces|generates|provides).*{object}',
                    'relation': 'produces'
                },
                {
                    'pattern': r'{object}.*(?:from|using|by).*{subject}',
                    'relation': 'produced_by'
                }
            ],
            ('Satellite', 'Location'): [
                {
                    'pattern': r'{subject}.*(?:covers|observes|monitors).*{object}',
                    'relation': 'observes'
                },
                {
                    'pattern': r'{object}.*(?:covered by|observed by|monitored by).*{subject}',
                    'relation': 'observed_by'
                }
            ],
            ('Product', 'Location'): [
                {
                    'pattern': r'{subject}.*(?:covers|for|of).*{object}',
                    'relation': 'covers'
                },
                {
                    'pattern': r'{object}.*(?:data|imagery|information).*{subject}',
                    'relation': 'covered_by'
                }
            ],
            ('Satellite', 'Date'): [
                {
                    'pattern': r'{subject}.*(?:launched|operational|active).*(?:on|in|since).*{object}',
                    'relation': 'launched_on'
                },
                {
                    'pattern': r'(?:on|in).*{object}.*{subject}.*(?:launched|became operational)',
                    'relation': 'launched_on'
                }
            ],
            ('Mission', 'Date'): [
                {
                    'pattern': r'{subject}.*(?:started|began|launched).*(?:on|in).*{object}',
                    'relation': 'started_on'
                },
                {
                    'pattern': r'(?:on|in).*{object}.*{subject}.*(?:started|began)',
                    'relation': 'started_on'
                }
            ]
        }
        
        # Dependency patterns for relation extraction
        self.dependency_patterns = [
            {
                'pattern': [
                    {'DEP': 'nsubj', 'TAG': 'NNP'},  # Subject
                    {'DEP': 'ROOT', 'POS': 'VERB'},   # Verb
                    {'DEP': 'dobj', 'TAG': 'NNP'}     # Object
                ],
                'relation_mapping': {
                    'operates': 'operates',
                    'uses': 'uses',
                    'provides': 'provides',
                    'supports': 'supports',
                    'carries': 'carries',
                    'monitors': 'monitors',
                    'observes': 'observes'
                }
            }
        ]
        
        # Preposition-based patterns
        self.preposition_patterns = [
            {
                'preposition': 'by',
                'relations': {
                    'operated': 'operated_by',
                    'developed': 'developed_by',
                    'launched': 'launched_by',
                    'managed': 'managed_by'
                }
            },
            {
                'preposition': 'on',
                'relations': {
                    'onboard': 'installed_on',
                    'mounted': 'mounted_on',
                    'carried': 'carried_on'
                }
            },
            {
                'preposition': 'from',
                'relations': {
                    'data': 'data_from',
                    'imagery': 'imagery_from',
                    'information': 'information_from'
                }
            }
        ]
    
    def _load_spacy_model(self):
        """Load spaCy model for dependency parsing."""
        try:
            model_name = config.get('models.ner_model', 'en_core_web_sm')
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model for relation extraction: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
            self.nlp = None
    
    def _generate_relation_id(self, subject: Entity, predicate: str, obj: Entity) -> str:
        """Generate unique relation ID."""
        content = f"{subject.id}_{predicate}_{obj.id}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _extract_pattern_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations using predefined patterns."""
        relations = []
        
        # Get entity pairs for each relation pattern
        for entity_types, patterns in self.relation_patterns.items():
            subject_type, object_type = entity_types
            
            # Find entities of the specified types
            subjects = [e for e in entities if e.label == subject_type]
            objects = [e for e in entities if e.label == object_type]
            
            # Check patterns for each entity pair
            for subject in subjects:
                for obj in objects:
                    if subject.id == obj.id:  # Skip self-relations
                        continue
                    
                    for pattern_config in patterns:
                        pattern = pattern_config['pattern']
                        relation_type = pattern_config['relation']
                        
                        # Replace placeholders with actual entity text
                        filled_pattern = pattern.format(
                            subject=re.escape(subject.text),
                            object=re.escape(obj.text)
                        )
                        
                        match = re.search(filled_pattern, text, re.IGNORECASE | re.DOTALL)
                        if match:
                            # Extract the context around the match
                            start_pos = max(0, match.start() - 50)
                            end_pos = min(len(text), match.end() + 50)
                            context = text[start_pos:end_pos]
                            
                            relation = Relation(
                                id=self._generate_relation_id(subject, relation_type, obj),
                                subject_entity=subject,
                                predicate=relation_type,
                                object_entity=obj,
                                confidence=0.8,  # Pattern-based relations have high confidence
                                context=context,
                                source_sentence=match.group(),
                                attributes={
                                    'extraction_method': 'pattern',
                                    'pattern': pattern,
                                    'match_position': (match.start(), match.end())
                                }
                            )
                            relations.append(relation)
        
        return relations
    
    def _extract_dependency_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations using dependency parsing."""
        relations = []
        
        if not self.nlp:
            return relations
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Create entity lookup by position
        entity_lookup = {}
        for entity in entities:
            for i in range(entity.start_pos, entity.end_pos):
                entity_lookup[i] = entity
        
        # Analyze dependencies
        for sent in doc.sents:
            # Find subject-verb-object patterns
            for token in sent:
                if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                    verb = token.lemma_.lower()
                    
                    # Find subject
                    subject_token = None
                    for child in token.children:
                        if child.dep_ in ['nsubj', 'nsubjpass']:
                            subject_token = child
                            break
                    
                    # Find object
                    object_token = None
                    for child in token.children:
                        if child.dep_ in ['dobj', 'pobj']:
                            object_token = child
                            break
                    
                    if subject_token and object_token:
                        # Check if tokens correspond to our entities
                        subject_entity = None
                        object_entity = None
                        
                        for entity in entities:
                            if (entity.start_pos <= subject_token.idx < entity.end_pos):
                                subject_entity = entity
                            if (entity.start_pos <= object_token.idx < entity.end_pos):
                                object_entity = entity
                        
                        if subject_entity and object_entity and subject_entity.id != object_entity.id:
                            # Map verb to relation type
                            relation_type = self._map_verb_to_relation(verb, subject_entity.label, object_entity.label)
                            
                            if relation_type:
                                relation = Relation(
                                    id=self._generate_relation_id(subject_entity, relation_type, object_entity),
                                    subject_entity=subject_entity,
                                    predicate=relation_type,
                                    object_entity=object_entity,
                                    confidence=0.7,  # Medium confidence for dependency parsing
                                    context=sent.text,
                                    source_sentence=sent.text,
                                    attributes={
                                        'extraction_method': 'dependency',
                                        'verb': verb,
                                        'dependency_path': f"{subject_token.dep_}-{token.dep_}-{object_token.dep_}"
                                    }
                                )
                                relations.append(relation)
        
        return relations
    
    def _map_verb_to_relation(self, verb: str, subject_type: str, object_type: str) -> Optional[str]:
        """Map verb and entity types to relation type."""
        verb_mappings = {
            'operate': 'operates',
            'use': 'uses',
            'carry': 'carries',
            'provide': 'provides',
            'support': 'supports',
            'monitor': 'monitors',
            'observe': 'observes',
            'launch': 'launches',
            'develop': 'develops',
            'manage': 'manages',
            'produce': 'produces',
            'generate': 'generates',
            'collect': 'collects',
            'process': 'processes'
        }
        
        base_relation = verb_mappings.get(verb)
        if not base_relation:
            return None
        
        # Adjust relation based on entity types
        if subject_type == 'Organization' and object_type == 'Satellite':
            if base_relation in ['operate', 'launch', 'develop', 'manage']:
                return base_relation
        elif subject_type == 'Satellite' and object_type == 'Technology':
            if base_relation in ['carry', 'use']:
                return base_relation
        elif subject_type == 'Technology' and object_type == 'Product':
            if base_relation in ['produce', 'generate']:
                return base_relation
        elif subject_type == 'Satellite' and object_type == 'Location':
            if base_relation in ['monitor', 'observe']:
                return base_relation
        
        return base_relation
    
    def _extract_preposition_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations based on preposition patterns."""
        relations = []
        
        if not self.nlp:
            return relations
        
        doc = self.nlp(text)
        
        # Create entity lookup
        entity_lookup = {}
        for entity in entities:
            entity_lookup[entity.text.lower()] = entity
        
        for sent in doc.sents:
            for prep_config in self.preposition_patterns:
                preposition = prep_config['preposition']
                relation_mappings = prep_config['relations']
                
                # Find preposition in sentence
                for token in sent:
                    if token.text.lower() == preposition and token.pos_ == 'ADP':
                        # Look for patterns around the preposition
                        for verb, relation_type in relation_mappings.items():
                            # Pattern: [Entity] [verb] [prep] [Entity]
                            pattern = rf'\b(\w+)\s+{re.escape(verb)}\w*\s+{re.escape(preposition)}\s+(\w+)\b'
                            matches = re.finditer(pattern, sent.text, re.IGNORECASE)
                            
                            for match in matches:
                                subject_text = match.group(1).lower()
                                object_text = match.group(2).lower()
                                
                                subject_entity = entity_lookup.get(subject_text)
                                object_entity = entity_lookup.get(object_text)
                                
                                if subject_entity and object_entity and subject_entity.id != object_entity.id:
                                    relation = Relation(
                                        id=self._generate_relation_id(subject_entity, relation_type, object_entity),
                                        subject_entity=subject_entity,
                                        predicate=relation_type,
                                        object_entity=object_entity,
                                        confidence=0.6,  # Lower confidence for preposition patterns
                                        context=sent.text,
                                        source_sentence=sent.text,
                                        attributes={
                                            'extraction_method': 'preposition',
                                            'preposition': preposition,
                                            'verb': verb
                                        }
                                    )
                                    relations.append(relation)
        
        return relations
    
    def _extract_proximity_relations(self, entities: List[Entity], text: str) -> List[Relation]:
        """Extract relations based on entity proximity in text."""
        relations = []
        
        # Sort entities by position
        sorted_entities = sorted(entities, key=lambda x: x.start_pos)
        
        # Look for entities that appear close to each other
        for i in range(len(sorted_entities)):
            for j in range(i + 1, len(sorted_entities)):
                entity1 = sorted_entities[i]
                entity2 = sorted_entities[j]
                
                # Calculate distance between entities
                distance = entity2.start_pos - entity1.end_pos
                
                # If entities are within 100 characters, consider them related
                if distance < 100 and distance > 0:
                    # Extract text between entities
                    between_text = text[entity1.end_pos:entity2.start_pos]
                    
                    # Determine relation type based on entity types and context
                    relation_type = self._infer_relation_from_context(
                        entity1, entity2, between_text
                    )
                    
                    if relation_type:
                        # Get broader context
                        start_pos = max(0, entity1.start_pos - 50)
                        end_pos = min(len(text), entity2.end_pos + 50)
                        context = text[start_pos:end_pos]
                        
                        relation = Relation(
                            id=self._generate_relation_id(entity1, relation_type, entity2),
                            subject_entity=entity1,
                            predicate=relation_type,
                            object_entity=entity2,
                            confidence=0.4,  # Low confidence for proximity-based relations
                            context=context,
                            source_sentence=between_text,
                            attributes={
                                'extraction_method': 'proximity',
                                'distance': distance,
                                'between_text': between_text.strip()
                            }
                        )
                        relations.append(relation)
        
        return relations
    
    def _infer_relation_from_context(self, entity1: Entity, entity2: Entity, context: str) -> Optional[str]:
        """Infer relation type from context between entities."""
        context_lower = context.lower().strip()
        
        # Common connecting words and their implied relations
        if any(word in context_lower for word in ['of', 'from', "'s"]):
            if entity1.label == 'Product' and entity2.label == 'Satellite':
                return 'from'
            elif entity1.label == 'Technology' and entity2.label == 'Satellite':
                return 'on'
        
        elif any(word in context_lower for word in ['and', ',', 'with']):
            # Entities mentioned together - general relation
            return 'related_to'
        
        elif any(word in context_lower for word in ['by', 'via', 'through']):
            if entity1.label == 'Satellite' and entity2.label == 'Organization':
                return 'operated_by'
            elif entity1.label == 'Product' and entity2.label == 'Technology':
                return 'produced_by'
        
        elif any(word in context_lower for word in ['in', 'at', 'over']):
            if entity2.label == 'Location':
                return 'located_at'
        
        # Default relation for entities that appear together
        if entity1.label != entity2.label:
            return 'related_to'
        
        return None
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate relations."""
        seen_relations = set()
        deduplicated = []
        
        for relation in relations:
            # Create a key for deduplication
            key = (relation.subject_entity.id, relation.predicate, relation.object_entity.id)
            
            if key not in seen_relations:
                seen_relations.add(key)
                deduplicated.append(relation)
            else:
                # If we see the same relation again, keep the one with higher confidence
                for i, existing in enumerate(deduplicated):
                    if (existing.subject_entity.id == relation.subject_entity.id and
                        existing.predicate == relation.predicate and
                        existing.object_entity.id == relation.object_entity.id):
                        
                        if relation.confidence > existing.confidence:
                            deduplicated[i] = relation
                        break
        
        return deduplicated
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        Extract all relations from text and entities.
        
        Args:
            text: Source text content
            entities: List of extracted entities
            
        Returns:
            List of extracted relations
        """
        all_relations = []
        
        # Extract using different methods
        pattern_relations = self._extract_pattern_relations(text, entities)
        dependency_relations = self._extract_dependency_relations(text, entities)
        preposition_relations = self._extract_preposition_relations(text, entities)
        proximity_relations = self._extract_proximity_relations(entities, text)
        
        # Combine all relations
        all_relations.extend(pattern_relations)
        all_relations.extend(dependency_relations)
        all_relations.extend(preposition_relations)
        all_relations.extend(proximity_relations)
        
        # Deduplicate
        final_relations = self._deduplicate_relations(all_relations)
        
        logger.info(f"Extracted {len(final_relations)} relations from {len(entities)} entities")
        return final_relations
    
    def batch_extract_relations(self, documents: List[Dict], entity_results: Dict[str, List[Entity]]) -> Dict[str, List[Relation]]:
        """
        Extract relations from multiple documents.
        
        Args:
            documents: List of document dictionaries
            entity_results: Dictionary mapping document IDs to their entities
            
        Returns:
            Dictionary mapping document IDs to their extracted relations
        """
        results = {}
        
        for doc in documents:
            document_id = doc.get('document_id', 'unknown')
            content = doc.get('content', '')
            
            entities = entity_results.get(document_id, [])
            
            if content and entities:
                relations = self.extract_relations(content, entities)
                results[document_id] = relations
            else:
                results[document_id] = []
        
        total_relations = sum(len(relations) for relations in results.values())
        logger.info(f"Extracted {total_relations} relations from {len(documents)} documents")
        
        return results
    
    def get_relation_statistics(self, relations: List[Relation]) -> Dict[str, Any]:
        """Generate statistics about extracted relations."""
        if not relations:
            return {}
        
        stats = {
            'total_relations': len(relations),
            'relation_types': {},
            'confidence_distribution': {
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0
            },
            'extraction_methods': {},
            'entity_type_pairs': {},
            'avg_confidence': sum(r.confidence for r in relations) / len(relations)
        }
        
        for relation in relations:
            # Count by relation type
            predicate = relation.predicate
            stats['relation_types'][predicate] = stats['relation_types'].get(predicate, 0) + 1
            
            # Count by confidence
            if relation.confidence > 0.7:
                stats['confidence_distribution']['high_confidence'] += 1
            elif relation.confidence > 0.4:
                stats['confidence_distribution']['medium_confidence'] += 1
            else:
                stats['confidence_distribution']['low_confidence'] += 1
            
            # Count by extraction method
            method = relation.attributes.get('extraction_method', 'unknown')
            stats['extraction_methods'][method] = stats['extraction_methods'].get(method, 0) + 1
            
            # Count by entity type pairs
            type_pair = f"{relation.subject_entity.label}-{relation.object_entity.label}"
            stats['entity_type_pairs'][type_pair] = stats['entity_type_pairs'].get(type_pair, 0) + 1
        
        return stats
    
    def save_relations(self, relations: List[Relation], output_file: str):
        """Save extracted relations to JSON file."""
        relation_dicts = []
        for relation in relations:
            relation_dict = {
                'id': relation.id,
                'subject': {
                    'id': relation.subject_entity.id,
                    'text': relation.subject_entity.text,
                    'label': relation.subject_entity.label
                },
                'predicate': relation.predicate,
                'object': {
                    'id': relation.object_entity.id,
                    'text': relation.object_entity.text,
                    'label': relation.object_entity.label
                },
                'confidence': relation.confidence,
                'context': relation.context,
                'source_sentence': relation.source_sentence,
                'attributes': relation.attributes
            }
            relation_dicts.append(relation_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(relation_dicts, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(relations)} relations to {output_file}")