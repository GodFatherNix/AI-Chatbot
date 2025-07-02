from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from datetime import datetime
from config import config
from ai_retrieval import AIRetrievalEngine
from knowledge_graph import KnowledgeGraphBuilder
from vector_db import EmbeddingManager
import uvicorn
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Help Bot API",
    description="AI-based Help bot for information retrieval from knowledge graphs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
ai_engine = None
knowledge_graph_builder = None
embedding_manager = None

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    sources: List[Dict]
    method: str
    timestamp: str

class BuildKnowledgeGraphRequest(BaseModel):
    urls: List[str]
    max_depth: Optional[int] = 2
    max_pages: Optional[int] = 100

class SystemStats(BaseModel):
    vector_database: Dict
    knowledge_graph: Dict
    ai_integration: bool
    status: str

@app.on_event("startup")
async def startup_event():
    """Initialize the AI system on startup"""
    global ai_engine, knowledge_graph_builder, embedding_manager
    
    logger.info("Initializing AI Help Bot system...")
    
    try:
        # Initialize components
        ai_engine = AIRetrievalEngine()
        knowledge_graph_builder = KnowledgeGraphBuilder()
        embedding_manager = EmbeddingManager()
        
        # Try to load existing knowledge graph
        try:
            knowledge_graph_builder.load_from_file(config.KNOWLEDGE_GRAPH_PATH)
            logger.info("Loaded existing knowledge graph")
        except Exception as e:
            logger.info(f"No existing knowledge graph found: {e}")
        
        logger.info("AI Help Bot system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI system: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Help Bot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query": "/query",
            "build_knowledge_graph": "/build-knowledge-graph",
            "stats": "/stats",
            "health": "/health"
        }
    }

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base and get AI-powered answers"""
    if not ai_engine:
        raise HTTPException(status_code=500, detail="AI engine not initialized")
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Process the query
        result = ai_engine.answer_query(request.query, request.max_results)
        
        response = QueryResponse(
            query=result['query'],
            answer=result['response']['answer'],
            confidence=result['response']['confidence'],
            sources=result['response']['sources'],
            method=result['response']['method'],
            timestamp=result['timestamp']
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/build-knowledge-graph")
async def build_knowledge_graph(request: BuildKnowledgeGraphRequest, background_tasks: BackgroundTasks):
    """Build knowledge graph from provided URLs"""
    if not knowledge_graph_builder or not embedding_manager:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Add the task to background
    background_tasks.add_task(
        _build_knowledge_graph_task,
        request.urls,
        request.max_depth,
        request.max_pages
    )
    
    return {
        "message": "Knowledge graph building started",
        "urls": request.urls,
        "max_depth": request.max_depth,
        "max_pages": request.max_pages,
        "status": "processing"
    }

async def _build_knowledge_graph_task(urls: List[str], max_depth: int, max_pages: int):
    """Background task to build knowledge graph"""
    try:
        logger.info(f"Starting knowledge graph build from {len(urls)} URLs")
        
        # Build knowledge graph
        knowledge_graph_builder.build_from_urls(urls, max_depth, max_pages)
        
        # Save the knowledge graph
        knowledge_graph_builder.save_to_file(config.KNOWLEDGE_GRAPH_PATH)
        
        # Build embeddings
        embedding_manager.build_embeddings_from_knowledge_graph(knowledge_graph_builder)
        
        # Update the AI engine's knowledge graph reference
        ai_engine.knowledge_graph = knowledge_graph_builder
        
        logger.info("Knowledge graph build completed successfully")
        
    except Exception as e:
        logger.error(f"Error building knowledge graph: {e}")

@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics and health information"""
    if not ai_engine:
        raise HTTPException(status_code=500, detail="AI engine not initialized")
    
    try:
        stats = ai_engine.get_stats()
        
        return SystemStats(
            vector_database=stats['vector_database'],
            knowledge_graph=stats['knowledge_graph'],
            ai_integration=stats['ai_integration'],
            status="healthy"
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "ai_engine": ai_engine is not None,
                "knowledge_graph": knowledge_graph_builder is not None,
                "embedding_manager": embedding_manager is not None
            }
        }
        
        # Check if vector database is accessible
        if ai_engine:
            try:
                vector_stats = ai_engine.vector_db.get_collection_stats()
                health_status["components"]["vector_database"] = True
                health_status["vector_db_document_count"] = vector_stats.get('document_count', 0)
            except:
                health_status["components"]["vector_database"] = False
        
        # Check if knowledge graph has data
        if knowledge_graph_builder:
            health_status["knowledge_graph_nodes"] = knowledge_graph_builder.graph.number_of_nodes()
            health_status["knowledge_graph_edges"] = knowledge_graph_builder.graph.number_of_edges()
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/search-history")
async def get_search_history():
    """Get recent search history (placeholder for future implementation)"""
    # This would typically connect to a database to store/retrieve search history
    return {
        "message": "Search history feature not yet implemented",
        "recent_queries": []
    }

@app.post("/feedback")
async def submit_feedback(feedback: Dict):
    """Submit feedback about query results"""
    # This would typically store feedback in a database for system improvement
    logger.info(f"Received feedback: {feedback}")
    
    return {
        "message": "Feedback received",
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/clear-knowledge-base")
async def clear_knowledge_base():
    """Clear the knowledge base (for testing/maintenance)"""
    if not ai_engine or not embedding_manager:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        # Clear vector database
        ai_engine.vector_db.clear_collection()
        
        # Clear knowledge graph
        knowledge_graph_builder.graph.clear()
        
        logger.info("Knowledge base cleared")
        
        return {
            "message": "Knowledge base cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing knowledge base: {str(e)}")

@app.get("/example-queries")
async def get_example_queries():
    """Get example queries for users to try"""
    examples = [
        {
            "query": "What is machine learning?",
            "description": "Get a definition of machine learning",
            "intent": "definition"
        },
        {
            "query": "How to implement a neural network?",
            "description": "Learn how to implement neural networks",
            "intent": "howto"
        },
        {
            "query": "List all available APIs",
            "description": "Get a list of available APIs",
            "intent": "list"
        },
        {
            "query": "Compare React vs Vue",
            "description": "Compare different technologies",
            "intent": "comparison"
        },
        {
            "query": "Fix authentication error",
            "description": "Troubleshoot common issues",
            "intent": "troubleshooting"
        }
    ]
    
    return {
        "examples": examples,
        "total": len(examples)
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info"
    )