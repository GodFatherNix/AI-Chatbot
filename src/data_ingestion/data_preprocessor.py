"""
Data preprocessor for cleaning and structuring extracted content.
"""
import re
import string
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import unicodedata

from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Could not download NLTK data")

@dataclass
class ProcessedDocument:
    """Represents a processed document with structured content."""
    document_id: str
    title: str
    content: str
    sentences: List[str]
    tokens: List[str]
    entities: List[Dict]
    keywords: List[str]
    language: str
    content_type: str
    metadata: Dict[str, Any]
    tables: List[Dict]
    images: List[Dict]
    links: List[str]
    sections: List[Dict]
    processing_timestamp: str

class DataPreprocessor:
    """
    Preprocesses extracted content for knowledge graph creation and model training.
    Handles text cleaning, normalization, entity extraction, and structuring.
    """
    
    def __init__(self):
        self.language = config.get('data_processing.language', 'en')
        self.max_text_length = config.get('data_processing.max_text_length', 8192)
        
        # Initialize NLP tools
        self.nlp = None
        self._load_spacy_model()
        
        # Initialize NLTK tools
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except:
            logger.warning("Could not initialize NLTK tools")
            self.lemmatizer = None
            self.stop_words = set()
        
        # MOSDAC-specific patterns
        self.satellite_patterns = [
            r'\b(?:INSAT|IRS|Oceansat|ResourceSat|CartoSat|RISAT|SARAL|Astrosat|Chandrayaan|Mangalyaan)\b',
            r'\b(?:Landsat|MODIS|AVHRR|VIIRS|Sentinel|SPOT)\b',
            r'\b(?:ISRO|NASA|ESA|NOAA)\b'
        ]
        
        self.geospatial_patterns = [
            r'\b\d+(?:\.\d+)?°?[NS]\s*,?\s*\d+(?:\.\d+)?°?[EW]\b',  # Coordinates
            r'\b(?:latitude|longitude|lat|lon|coord)\b',
            r'\b(?:WGS84|UTM|geographic|projection)\b',
            r'\b(?:orbit|swath|resolution|pixel|band)\b'
        ]
        
        self.temporal_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # Date patterns
            r'\b\d{2}/\d{2}/\d{4}\b',
            r'\b(?:daily|weekly|monthly|yearly|annual)\b',
            r'\b(?:pass|overpass|acquisition)\b'
        ]
    
    def _load_spacy_model(self):
        """Load spaCy NLP model."""
        try:
            model_name = config.get('models.ner_model', 'en_core_web_sm')
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
            self.nlp = None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters except newlines and tabs
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # Fix common encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        text = text.replace('â€"', '-')
        text = text.replace('â€"', '–')
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
        
        # Clean up punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\'\/]', ' ', text)
        
        # Normalize spacing around punctuation
        text = re.sub(r'\s*([\.!?;:,])\s*', r'\1 ', text)
        text = re.sub(r'\s*([()"])\s*', r' \1 ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _extract_sections(self, text: str) -> List[Dict]:
        """Extract document sections based on headers and structure."""
        sections = []
        
        # Split by common section patterns
        section_patterns = [
            r'\n(?=\d+\.?\s+[A-Z][^.!?]*(?:\n|$))',  # Numbered sections
            r'\n(?=[A-Z][A-Z\s]+:)',  # ALL CAPS headers with colon
            r'\n(?=[A-Z][^.!?]*(?:\n|$))',  # Title case headers
        ]
        
        current_section = {"title": "Introduction", "content": "", "start_pos": 0}
        
        # Simple section detection
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if line looks like a header
            if (len(line) > 0 and len(line) < 100 and 
                (line.isupper() or line.istitle()) and 
                not line.endswith('.') and 
                ':' in line or line.endswith(':')):
                
                # Save previous section
                if current_section["content"].strip():
                    sections.append(current_section.copy())
                
                # Start new section
                current_section = {
                    "title": line.replace(':', '').strip(),
                    "content": "",
                    "start_pos": i
                }
            else:
                current_section["content"] += line + "\n"
        
        # Add last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections = [{
                "title": "Content",
                "content": text,
                "start_pos": 0
            }]
        
        return sections
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities using spaCy and custom patterns."""
        entities = []
        
        if self.nlp:
            doc = self.nlp(text[:1000000])  # Limit text length for spaCy
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 1.0,
                    "source": "spacy"
                })
        
        # Custom pattern matching for MOSDAC-specific entities
        patterns = {
            "SATELLITE": self.satellite_patterns,
            "COORDINATE": self.geospatial_patterns,
            "DATE": self.temporal_patterns
        }
        
        for label, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        "text": match.group(),
                        "label": label,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.8,
                        "source": "pattern"
                    })
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x["start"])
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities based on text and position overlap."""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x["start"])
        
        deduplicated = [entities[0]]
        
        for entity in entities[1:]:
            overlap = False
            for existing in deduplicated:
                # Check for significant overlap
                overlap_start = max(entity["start"], existing["start"])
                overlap_end = min(entity["end"], existing["end"])
                overlap_length = max(0, overlap_end - overlap_start)
                
                entity_length = entity["end"] - entity["start"]
                existing_length = existing["end"] - existing["start"]
                
                if (overlap_length > 0.5 * min(entity_length, existing_length) or
                    entity["text"].lower() == existing["text"].lower()):
                    overlap = True
                    break
            
            if not overlap:
                deduplicated.append(entity)
        
        return deduplicated
    
    def _extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract important keywords from text."""
        if not text:
            return []
        
        # Clean and tokenize
        cleaned_text = self._clean_text(text)
        words = word_tokenize(cleaned_text.lower())
        
        # Remove stopwords and short words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words and 
            len(word) > 2 and 
            word.isalpha()
        ]
        
        # Lemmatize if available
        if self.lemmatizer:
            filtered_words = [self.lemmatizer.lemmatize(word) for word in filtered_words]
        
        # Count word frequencies
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords]]
    
    def _process_tables(self, tables: List[Dict]) -> List[Dict]:
        """Process and clean table data."""
        processed_tables = []
        
        for table in tables:
            if not table.get('data'):
                continue
            
            try:
                # Convert to DataFrame for processing
                df = pd.DataFrame(table['data'])
                
                # Clean data
                df = df.replace('', np.nan)
                df = df.dropna(how='all')  # Remove empty rows
                df = df.dropna(axis=1, how='all')  # Remove empty columns
                
                # Try to detect header row
                if len(df) > 1:
                    # Check if first row looks like headers
                    first_row = df.iloc[0]
                    if all(isinstance(val, str) and not val.isdigit() for val in first_row if pd.notna(val)):
                        df.columns = first_row
                        df = df.drop(df.index[0])
                
                processed_table = {
                    'table_id': table.get('table_id', len(processed_tables)),
                    'data': df.values.tolist(),
                    'columns': df.columns.tolist() if hasattr(df.columns, 'tolist') else list(range(len(df.columns))),
                    'rows': len(df),
                    'column_count': len(df.columns),
                    'metadata': {
                        'has_headers': hasattr(df.columns, 'tolist'),
                        'numeric_columns': [col for col in df.columns if df[col].dtype in ['int64', 'float64']],
                        'text_columns': [col for col in df.columns if df[col].dtype == 'object']
                    }
                }
                
                processed_tables.append(processed_table)
                
            except Exception as e:
                logger.warning(f"Error processing table: {e}")
                processed_tables.append(table)  # Keep original if processing fails
        
        return processed_tables
    
    def process_document(self, content: Dict) -> ProcessedDocument:
        """
        Process a single document from extracted content.
        
        Args:
            content: Dictionary containing extracted content
            
        Returns:
            ProcessedDocument with structured and cleaned content
        """
        # Extract basic information
        document_id = content.get('file_hash', '')
        title = content.get('title', 'Untitled')
        raw_text = content.get('text_content', '')
        content_type = content.get('content_type', 'unknown')
        metadata = content.get('metadata', {})
        tables = content.get('tables', [])
        images = content.get('images', [])
        links = content.get('links', [])
        
        # Clean and process text
        cleaned_text = self._clean_text(raw_text)
        
        # Truncate if too long
        if len(cleaned_text) > self.max_text_length:
            cleaned_text = cleaned_text[:self.max_text_length] + "..."
            logger.warning(f"Text truncated to {self.max_text_length} characters")
        
        # Extract sentences
        try:
            sentences = sent_tokenize(cleaned_text)
        except:
            sentences = cleaned_text.split('.')
        
        # Extract tokens
        try:
            tokens = word_tokenize(cleaned_text.lower())
        except:
            tokens = cleaned_text.lower().split()
        
        # Extract entities
        entities = self._extract_entities(cleaned_text)
        
        # Extract keywords
        keywords = self._extract_keywords(cleaned_text)
        
        # Extract sections
        sections = self._extract_sections(cleaned_text)
        
        # Process tables
        processed_tables = self._process_tables(tables)
        
        # Detect language (simple heuristic)
        language = self.language
        if any(ord(char) > 127 for char in cleaned_text[:1000]):
            language = 'unknown'
        
        # Update metadata with processing info
        metadata.update({
            'sentence_count': len(sentences),
            'token_count': len(tokens),
            'entity_count': len(entities),
            'keyword_count': len(keywords),
            'section_count': len(sections),
            'character_count': len(cleaned_text),
            'word_count': len(cleaned_text.split()),
            'processed_timestamp': datetime.now().isoformat()
        })
        
        return ProcessedDocument(
            document_id=document_id,
            title=title,
            content=cleaned_text,
            sentences=sentences,
            tokens=tokens,
            entities=entities,
            keywords=keywords,
            language=language,
            content_type=content_type,
            metadata=metadata,
            tables=processed_tables,
            images=images,
            links=links,
            sections=sections,
            processing_timestamp=datetime.now().isoformat()
        )
    
    def batch_process(self, content_list: List[Dict]) -> List[ProcessedDocument]:
        """Process multiple documents in batch."""
        processed_docs = []
        
        for i, content in enumerate(content_list):
            try:
                logger.info(f"Processing document {i+1}/{len(content_list)}")
                processed_doc = self.process_document(content)
                processed_docs.append(processed_doc)
            except Exception as e:
                logger.error(f"Error processing document {i+1}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_docs)} out of {len(content_list)} documents")
        return processed_docs
    
    def save_processed_documents(self, documents: List[ProcessedDocument], 
                               output_dir: str = "data/processed/documents"):
        """Save processed documents to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for doc in documents:
            filename = f"processed_{doc.document_id}_{int(datetime.now().timestamp())}.json"
            file_path = output_path / filename
            
            doc_dict = {
                'document_id': doc.document_id,
                'title': doc.title,
                'content': doc.content,
                'sentences': doc.sentences,
                'tokens': doc.tokens,
                'entities': doc.entities,
                'keywords': doc.keywords,
                'language': doc.language,
                'content_type': doc.content_type,
                'metadata': doc.metadata,
                'tables': doc.tables,
                'images': doc.images,
                'links': doc.links,
                'sections': doc.sections,
                'processing_timestamp': doc.processing_timestamp
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(documents)} processed documents to {output_path}")
    
    def get_processing_statistics(self, documents: List[ProcessedDocument]) -> Dict:
        """Generate statistics about processed documents."""
        if not documents:
            return {}
        
        stats = {
            'total_documents': len(documents),
            'total_sentences': sum(len(doc.sentences) for doc in documents),
            'total_tokens': sum(len(doc.tokens) for doc in documents),
            'total_entities': sum(len(doc.entities) for doc in documents),
            'total_keywords': sum(len(doc.keywords) for doc in documents),
            'content_types': {},
            'languages': {},
            'entity_types': {},
            'avg_document_length': sum(len(doc.content) for doc in documents) / len(documents),
            'avg_sentences_per_doc': sum(len(doc.sentences) for doc in documents) / len(documents),
            'avg_entities_per_doc': sum(len(doc.entities) for doc in documents) / len(documents)
        }
        
        # Count by content type
        for doc in documents:
            content_type = doc.content_type
            stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
            
            # Count by language
            language = doc.language
            stats['languages'][language] = stats['languages'].get(language, 0) + 1
            
            # Count entity types
            for entity in doc.entities:
                entity_type = entity['label']
                stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
        
        return stats