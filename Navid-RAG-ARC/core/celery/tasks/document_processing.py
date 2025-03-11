import os
import json
from celery import shared_task, states, chain
from celery.utils.log import get_task_logger
from core.celery.config import celery_app
import time
import redis
from core.settings import get_settings

logger = get_task_logger(__name__)
settings = get_settings()

def send_status_update(user_id, data):
    """
    Send status updates to clients via Redis pub/sub
    Synchronous version for Celery tasks
    """
    try:
        # Skip if no user_id provided
        if not user_id:
            logger.warning("No user_id provided, skipping status update")
            return
            
        # Create a channel specific to the user
        channel = f"user:{user_id}:tasks"
        
        # Use synchronous Redis client for Celery tasks
        r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        
        # Add timestamp to data
        from datetime import datetime
        data["timestamp"] = datetime.now().isoformat()
        
        message = json.dumps({
            "type": "task_update",
            "data": data
        })
        
        # Publish message
        result = r.publish(channel, message)
        
        # Log success or failure
        if result > 0:
            logger.info(f"Published event to {channel}: {data['status']} (received by {result} subscribers)")
        else:
            logger.warning(f"Published event to {channel}: {data['status']} (no subscribers)")
            
    except Exception as e:
        logger.error(f"Error sending status update: {e}")

@shared_task(bind=True, name="process_document")
def process_document(self, file_path, doc_id, conversation_id, user_id=None):
    """Process document and update status in real-time"""
    try:
        logger.info(f"Starting document processing for doc_id={doc_id}, user_id={user_id}")
        
        # Update status: started processing
        self.update_state(
            state=states.STARTED,
            meta={
                'status': 'PROCESSING',
                'message': 'Started processing document',
                'progress': 10,
                'doc_id': doc_id
            }
        )
        
        # Send status update via Redis
        send_status_update(user_id, {
            'task_id': self.request.id,
            'status': 'PROCESSING',
            'message': 'Started processing document',
            'progress': 10,
            'doc_id': doc_id
        })
        
        # Ensure file exists
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            send_status_update(user_id, {
                'task_id': self.request.id,
                'status': 'ERROR',
                'message': error_msg,
                'progress': 0,
                'doc_id': doc_id
            })
            raise FileNotFoundError(error_msg)
        
        # Update status: parsing document
        logger.info(f"Parsing document {doc_id}")
        self.update_state(
            state=states.STARTED,
            meta={
                'status': 'PARSING',
                'message': 'Parsing document content',
                'progress': 30,
                'doc_id': doc_id
            }
        )
        
        send_status_update(user_id, {
            'task_id': self.request.id,
            'status': 'PARSING',
            'message': 'Parsing document content',
            'progress': 30,
            'doc_id': doc_id
        })

        # Document processing logic here
        # (replace with your actual document processing code)
        time.sleep(1)  # Simulating processing time
        
        # Prepare document chunks
        chunks = [{"content": "Sample content", "metadata": {"source": file_path}, "context": ""}]
        logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
        
        # Send status before chaining to next task
        send_status_update(user_id, {
            'task_id': self.request.id,
            'status': 'CHUNKING_COMPLETE',
            'message': 'Document chunking complete, generating embeddings',
            'progress': 50,
            'doc_id': doc_id
        })
        
        # Chain to embedding generation task
        result = chain(
            generate_embeddings.s(chunks, doc_id, conversation_id, user_id, self.request.id)
        ).apply_async()
        
        logger.info(f"Chained to generate_embeddings task with ID: {result.id}")
        
        return {
            'status': 'CHUNKING_COMPLETE',
            'message': 'Document chunking complete, generating embeddings',
            'progress': 50,
            'doc_id': doc_id,
            'next_task_id': result.id
        }
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        send_status_update(user_id, {
            'task_id': self.request.id,
            'status': 'ERROR',
            'message': f'Error processing document: {str(e)}',
            'progress': 0,
            'doc_id': doc_id
        })
        raise

@shared_task(bind=True, name="generate_embeddings")
def generate_embeddings(self, chunks, doc_id, conversation_id, user_id=None, parent_task_id=None):
    """Generate embeddings for document chunks"""
    try:
        logger.info(f"Generating embeddings for doc_id={doc_id}, user_id={user_id}")
        
        # Update status: generating embeddings
        self.update_state(
            state=states.STARTED,
            meta={
                'status': 'EMBEDDING',
                'message': 'Generating embeddings',
                'progress': 60,
                'doc_id': doc_id,
                'parent_task_id': parent_task_id
            }
        )
        
        send_status_update(user_id, {
            'task_id': self.request.id,
            'parent_task_id': parent_task_id,
            'status': 'EMBEDDING',
            'message': 'Generating embeddings',
            'progress': 60,
            'doc_id': doc_id
        })

        # Use the hybrid search index for indexing
        from core.search.hybrid_search import HybridSearchIndex
        search_index = HybridSearchIndex()
        search_index.index_chunks(chunks, doc_id, conversation_id)
        logger.info(f"Indexed chunks for document {doc_id}")
        
        # Send progress updates
        send_status_update(user_id, {
            'task_id': self.request.id,
            'parent_task_id': parent_task_id,
            'status': 'INDEXING',
            'message': 'Indexing document chunks',
            'progress': 90,
            'doc_id': doc_id
        })
        
        # Short delay to ensure status updates are processed in order
        time.sleep(0.5)
        
        # Update status: completed
        send_status_update(user_id, {
            'task_id': self.request.id,
            'parent_task_id': parent_task_id,
            'status': 'COMPLETE',
            'message': 'Document processing complete',
            'progress': 100,
            'doc_id': doc_id
        })
        
        logger.info(f"Completed embedding generation for document {doc_id}")
        
        return {
            'status': 'COMPLETE',
            'message': 'Document processing complete',
            'progress': 100,
            'doc_id': doc_id
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        send_status_update(user_id, {
            'task_id': self.request.id,
            'parent_task_id': parent_task_id,
            'status': 'ERROR',
            'message': f'Error generating embeddings: {str(e)}',
            'progress': 0,
            'doc_id': doc_id
        })
        raise
