"""
Embeddings generation tasks
This module contains Celery tasks for generating embeddings
"""
from celery import shared_task, states
from celery.utils.log import get_task_logger
import time
from core.settings import get_settings

logger = get_task_logger(__name__)
settings = get_settings()

@shared_task(bind=True, name="generate_embeddings_async")
def generate_embeddings_async(self, texts, metadata=None, user_id=None):
    """
    Generate embeddings for a list of texts asynchronously
    
    This task is used when embeddings need to be generated independently
    of document processing, such as for queries or pre-computed embeddings.
    """
    try:
        # Update status
        self.update_state(
            state=states.STARTED,
            meta={
                'status': 'EMBEDDING',
                'message': 'Generating embeddings',
                'progress': 10,
                'count': len(texts)
            }
        )
        
        # Publish status to Redis if user_id is provided
        if user_id:
            from core.celery.tasks.document_processing import send_status_update
            send_status_update(user_id, {
                'task_id': self.request.id,
                'status': 'EMBEDDING',
                'message': f'Generating embeddings for {len(texts)} texts',
                'progress': 10
            })
        
        # Get embeddings from settings
        embeddings_model = settings.get_embeddings()
        
        # Process in batches if needed
        batch_size = 10
        results = []
        
        # Simulate progress for each batch
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Here we would actually generate embeddings
            # For now, just simulate processing time
            time.sleep(0.5)
            
            # In a real implementation, we would do:
            # batch_embeddings = embeddings_model.embed_documents(batch)
            # results.extend(batch_embeddings)
            
            # Update progress
            progress = min(90, int(10 + (i + len(batch)) / len(texts) * 80))
            self.update_state(
                state=states.STARTED,
                meta={
                    'status': 'EMBEDDING',
                    'message': f'Generated {i + len(batch)}/{len(texts)} embeddings',
                    'progress': progress,
                    'count': len(texts)
                }
            )
            
            if user_id:
                from core.celery.tasks.document_processing import send_status_update
                send_status_update(user_id, {
                    'task_id': self.request.id,
                    'status': 'EMBEDDING',
                    'message': f'Generated {i + len(batch)}/{len(texts)} embeddings',
                    'progress': progress
                })
        
        # In a real implementation, return the embeddings
        # return results
        
        # For now, just return success
        return {
            'status': 'COMPLETE',
            'message': 'Embedding generation complete',
            'progress': 100,
            'count': len(texts)
        }
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        if user_id:
            from core.celery.tasks.document_processing import send_status_update
            send_status_update(user_id, {
                'task_id': self.request.id,
                'status': 'ERROR',
                'message': f'Error generating embeddings: {str(e)}',
                'progress': 0
            })
        raise
