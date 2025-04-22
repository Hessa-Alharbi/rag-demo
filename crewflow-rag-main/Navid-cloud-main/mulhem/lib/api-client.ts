import axios from 'axios';

// Determine the base URL based on the environment
const getBaseUrl = () => {
  // تغيير عنوان الخادم ليشير إلى خدمة Navid-RAG-ARC
  return 'http://localhost:8000';
};

const apiClient = axios.create({
  baseURL: getBaseUrl(),
  headers: {
    'Accept': 'application/json',
  },
  timeout: 300000, // 5 minute timeout for large files
});

// Add proper typing for request interceptors
apiClient.interceptors.request.use(
  (config) => {
    // For form data requests, don't set Content-Type - browser will set it
    if (config.data instanceof FormData) {
      if (config.headers) {
        delete config.headers['Content-Type'];
      }
      console.log('FormData detected, removing Content-Type to let browser set it');
    } else {
      // For JSON requests, set the Content-Type
      if (config.headers) {
        config.headers['Content-Type'] = 'application/json';
      }
    }
    
    // Log requests for debugging
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`);
    
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Intercept responses to handle errors
apiClient.interceptors.response.use(
  (response) => {
    // Log successful responses
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    // Format error response for easier handling
    let errorMessage = 'An unknown error occurred';
    let statusCode = 500;
    
    if (error.response) {
      // The request was made and the server responded with an error status
      errorMessage = error.response.data?.detail || 
                    error.response.data?.message || 
                    `Server error: ${error.response.status}`;
      statusCode = error.response.status;
      
      console.error(`API Error ${statusCode}:`, error.response.data);
    } else if (error.request) {
      // The request was made but no response was received
      errorMessage = 'No response from server. Check your network connection.';
      statusCode = 0;
      console.error('API Network Error:', error.message);
    } else {
      // Something happened in setting up the request
      errorMessage = error.message || 'Error setting up API request';
      console.error('API Setup Error:', error.message);
    }
    
    return Promise.reject({
      message: errorMessage,
      status: statusCode,
      data: error.response?.data,
      original: error
    });
  }
);

export default apiClient;
