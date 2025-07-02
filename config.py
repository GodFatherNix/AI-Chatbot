import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the AI Help Bot system"""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # File paths
    KNOWLEDGE_GRAPH_PATH: str = os.getenv("KNOWLEDGE_GRAPH_PATH", "./data/knowledge_graph.json")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/chroma_db")
    
    # Web scraping settings
    WEB_SCRAPING_DELAY: float = float(os.getenv("WEB_SCRAPING_DELAY", "1.0"))
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", "10000"))
    
    # Server settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # Model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    @classmethod
    def ensure_data_dirs(cls):
        """Create necessary data directories"""
        os.makedirs(os.path.dirname(cls.KNOWLEDGE_GRAPH_PATH), exist_ok=True)
        os.makedirs(cls.VECTOR_DB_PATH, exist_ok=True)

# Global config instance
config = Config()
config.ensure_data_dirs()