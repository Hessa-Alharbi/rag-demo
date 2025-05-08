import { useState, useRef } from 'react';
import apiClient from '@/lib/api-client';
import axios from 'axios';

export interface UploadError {
  message: string;
  status: number;
  details?: any;
}

export interface UploadResponse {
  message: string;
  task_id: string;
  file_path: string;
  session_id?: string;
}

export const useFileUpload = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<UploadError | null>(null);
  const [uploadResponse, setUploadResponse] = useState<UploadResponse | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);

  const uploadQueue = useRef<Array<() => Promise<any>>>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const processQueue = async () => {
    if (isProcessing || uploadQueue.current.length === 0) return;
    
    setIsProcessing(true);
    try {
      const nextTask = uploadQueue.current.shift();
      if (nextTask) await nextTask();
    } finally {
      setIsProcessing(false);
      if (uploadQueue.current.length > 0) {
        setTimeout(processQueue, 1000); // انتظر ثانية بين الطلبات
      }
    }
  };

  // إضافة وظيفة إعادة المحاولة مع انتظار متزايد
  const retryWithBackoff = async (fn: () => Promise<any>, maxRetries = 3, initialDelay = 1000) => {
    let retries = 0;
    while (retries < maxRetries) {
      try {
        return await fn();
      } catch (error: any) {
        // إذا كان الخطأ مرتبط بالشبكة أو خطأ 5xx من الخادم، حاول مرة أخرى
        const isNetworkError = !error.response || error.code === 'ECONNABORTED';
        const isServerError = error.response?.status >= 500;
        if ((isNetworkError || isServerError) && retries < maxRetries - 1) {
          retries++;
          const delay = initialDelay * Math.pow(2, retries);
          console.log(`Retry attempt ${retries}/${maxRetries} after ${delay}ms`);
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
        throw error;
      }
    }
  };

  const uploadFile = async (file: File) => {
    return new Promise((resolve, reject) => {
      const task = async () => {
        setIsLoading(true);
        setError(null);
        setUploadResponse(null);
        setUploadProgress(0);

        const formData = new FormData();
        formData.append('file', file);

        // تحقق من نوع وحجم الملف قبل المحاولة
        const maxSize = 10 * 1024 * 1024; // 10 ميجابايت
        if (file.size > maxSize) {
          setError({
            message: `الملف كبير جدًا. الحد الأقصى المسموح به هو ${maxSize / (1024 * 1024)} ميجابايت`,
            status: 413
          });
          setIsLoading(false);
          throw new Error("File too large");
        }

        // الأنواع المدعومة
        const supportedTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
        if (!supportedTypes.includes(file.type) && !file.name.endsWith('.pdf') && !file.name.endsWith('.docx') && !file.name.endsWith('.doc') && !file.name.endsWith('.txt')) {
          console.warn(`File type might not be supported: ${file.type}`);
        }

        try {
          console.log('Starting file upload...');
          console.log('Uploading file:', file.name);
          
          // استخدام استدعاء مباشر لـ axios مع خيارات محسنة
          const result = await retryWithBackoff(async () => {
            const response = await apiClient.post<UploadResponse>('/initialize', formData, {
              headers: {
                'Content-Type': 'multipart/form-data'
              },
              timeout: 600000, // 10 minute timeout
              onUploadProgress: (progressEvent) => {
                if (progressEvent.total) {
                  const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                  setUploadProgress(progress);
                  console.log(`Upload progress: ${progress}%`);
                }
              }
            });
            
            const data = response.data;
            console.log('Upload response:', data);
            
            // Validate the response
            if (!data.task_id && !data.session_id) {
              throw new Error('Server did not return a valid task or session ID');
            }
            
            setUploadResponse(data);
            return data;
          }, 3, 2000);
          resolve(result);
        } catch (err: any) {
          console.error('Upload error:', err);
          
          // Provide more detailed error info
          let errorDetail = 'File upload failed';
          let errorStatus = 500;
          
          if (err.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            errorDetail = err.response.data?.detail || err.response.data?.message || err.message || 'Server error';
            errorStatus = err.response.status;
            console.error('Server response:', err.response.data);
          } else if (err.request) {
            // The request was made but no response was received
            errorDetail = 'No response from server. Check your network connection.';
            errorStatus = 0;
          } else {
            // Something happened in setting up the request
            errorDetail = err.message || 'Error setting up request';
          }
          
          const error = {
            message: errorDetail,
            status: errorStatus,
            details: err.response?.data
          };
          
          setError(error);
          reject(error);
        } finally {
          setIsLoading(false);
        }
      };
      
      uploadQueue.current.push(task);
      processQueue();
    });
  };

  return {
    uploadFile,
    isLoading,
    error,
    uploadResponse,
    uploadProgress,
    resetState: () => {
      setError(null);
      setUploadResponse(null);
      setUploadProgress(0);
    }
  };
};
