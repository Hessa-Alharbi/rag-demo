import { useState } from 'react';
import apiClient from '@/lib/api-client';

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

  const uploadFile = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setUploadResponse(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log('Uploading file:', file.name);
      
      // Try to use the correct endpoint for the server
      let response;
      try {
        // First try using the initialize endpoint
        response = await apiClient.post<UploadResponse>('/initialize', formData, {
          timeout: 600000, // 10 minute timeout
        });
      } catch (initError: any) {
        if (initError.status === 404) {
          // If the endpoint doesn't exist, try the attachments endpoint
          console.log('Endpoint /initialize not found, trying attachments endpoint');
          response = await apiClient.post<UploadResponse>('/conversations/new/attachments', formData, {
            timeout: 600000, // 10 minute timeout
          });
        } else {
          // Rethrow the original error if it's not a 404
          throw initError;
        }
      }
      
      const data = response.data;
      console.log('Upload response:', data);
      
      // Validate the response
      if (!data.task_id && !data.session_id) {
        throw new Error('Server did not return a valid task or session ID');
      }
      
      setUploadResponse(data);
      return data;
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
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  return {
    uploadFile,
    isLoading,
    error,
    uploadResponse,
    resetState: () => {
      setError(null);
      setUploadResponse(null);
    }
  };
};
