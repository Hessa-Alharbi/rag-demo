import os
from fastapi import UploadFile
import aiofiles
from uuid import UUID
import mimetypes
import hashlib
from datetime import datetime
from core.config import get_settings

settings = get_settings()

ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.doc', '.docx', '.csv', '.xls', '.xlsx']

class FileHandler:
    @staticmethod
    def create_upload_dir(conversation_id: UUID) -> str:
        # Create year/month based directory structure
        date_path = datetime.now().strftime("%Y/%m")
        dir_path = os.path.join(settings.UPLOAD_DIR, str(conversation_id), date_path)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    @staticmethod
    def get_safe_filename(filename: str) -> str:
        # Generate unique filename while preserving extension
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_hex = hashlib.md5(f"{name}{timestamp}".encode()).hexdigest()[:8]
        return f"{name}_{timestamp}_{random_hex}{ext}"

    @staticmethod
    def is_valid_file(filename: str, filesize: int) -> bool:
        if not filename:
            return False
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False
        if filesize and filesize > settings.MAX_FILE_SIZE:
            return False
        return True

    @staticmethod
    async def save_file(file: UploadFile, conversation_id: UUID) -> tuple[str, int]:
        dir_path = FileHandler.create_upload_dir(conversation_id)
        safe_filename = FileHandler.get_safe_filename(file.filename)
        file_path = os.path.join(dir_path, safe_filename)
        
        size = 0
        chunk_size = 1024 * 64  # 64KB chunks
        
        try:
            async with aiofiles.open(file_path, 'wb') as out_file:
                while chunk := await file.read(chunk_size):
                    size += len(chunk)
                    if size > settings.MAX_FILE_SIZE:
                        # Clean up partial file
                        await out_file.close()
                        os.remove(file_path)
                        raise ValueError(f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE/1024/1024}MB")
                    await out_file.write(chunk)
            
            return file_path, size
            
        except Exception as e:
            # Clean up in case of error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

    @staticmethod
    def get_mime_type(filename: str) -> str:
        mime_type = mimetypes.guess_type(filename)[0]
        return mime_type if mime_type else 'application/octet-stream'
