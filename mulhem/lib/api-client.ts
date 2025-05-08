import axios, { AxiosError, InternalAxiosRequestConfig } from 'axios';

// تعريف نوع موسع للطلبات يتضمن خصائص إعادة المحاولة
interface ExtendedAxiosRequestConfig extends InternalAxiosRequestConfig {
  retryCount?: number;
  maxRetries?: number;
  _retry?: boolean;
}

// تحديد عنوان API ثابت للباك إند
export function getBaseUrl() {
  // Use the environment variable if available
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  
  // Fallback to local development URL
  if (process.env.NODE_ENV === 'development') {
    return 'http://localhost:8000';
  }
  
  // Production fallback - أستخدم عنوان Railway الخاص بك هنا
  return 'https://ingenious-transformation-production-be7c.up.railway.app';;
};

// Create an Axios instance with a base URL
const apiClient = axios.create({
  baseURL: getBaseUrl(),
  timeout: 600000, // 10 minutes timeout for large file uploads
});

// سجل عنوان URL الأساسي عند إنشاء العميل
console.log('API client created with baseURL:', apiClient.defaults.baseURL);

// Add proper typing for request interceptors
apiClient.interceptors.request.use(
  (config: ExtendedAxiosRequestConfig) => {
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
    
    // Add Authorization header with token if available
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('access_token');
      if (token && config.headers) {
        config.headers['Authorization'] = `Bearer ${token}`;
        console.log('Adding auth token to request');
      }
    }
    
    // إضافة معلومات إعادة المحاولة للطلبات
    config.retryCount = config.retryCount || 0;
    config.maxRetries = 3; // عدد المحاولات الأقصى
    
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
  async (error: AxiosError) => {
    const originalRequest = error.config as ExtendedAxiosRequestConfig;
    
    // Check if error is due to authorization (401) and we haven't tried refreshing yet
    if (error.response?.status === 401 && 
        !originalRequest._retry && 
        typeof window !== 'undefined') {
      
      console.log('Token expired, attempting to refresh...');
      originalRequest._retry = true;
      
      try {
        // Get refresh token from storage
        const refreshToken = localStorage.getItem('refresh_token');
        if (!refreshToken) {
          console.error('No refresh token available');
          // Redirect to login page if no refresh token
          window.location.href = '/auth/login';
          return Promise.reject(error);
        }

        // Call the refresh endpoint
        const response = await axios.post(`${getBaseUrl()}/auth/refresh`, {
          refresh_token: refreshToken
        });
        
        if (response.data?.access_token) {
          // Update tokens in storage
          localStorage.setItem('access_token', response.data.access_token);
          localStorage.setItem('refresh_token', response.data.refresh_token);
          
          // Update authorization header in the original request
          originalRequest.headers['Authorization'] = `Bearer ${response.data.access_token}`;
    
          // Retry the original request with the new token
          return axios(originalRequest);
        }
      } catch (refreshError) {
        console.error('Failed to refresh token:', refreshError);
        
        // Clear tokens and redirect to login on refresh failure
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/auth/login';
        return Promise.reject(error);
      }
    }
    
    // إعادة المحاولة للأخطاء المتعلقة بالشبكة
    if (originalRequest && 
        !originalRequest._retry && 
        (error.code === 'ECONNABORTED' || error.message.includes('timeout') || !error.response)) {
      
      // إذا لم يتم تجاوز الحد الأقصى من المحاولات
      if (!originalRequest.maxRetries || originalRequest.retryCount! < originalRequest.maxRetries) {
        // زيادة عدد المحاولات
        originalRequest.retryCount = (originalRequest.retryCount || 0) + 1;
        
        // انتظر قبل إعادة المحاولة (زيادة الوقت مع كل محاولة)
        const delay = 1000 * Math.pow(2, originalRequest.retryCount);
        console.log(`Retrying request (${originalRequest.retryCount}/${originalRequest.maxRetries}) after ${delay}ms`);
        
        return new Promise(resolve => {
          setTimeout(() => resolve(apiClient(originalRequest)), delay);
        });
      }
    }
    
    // Format error response for easier handling
    let errorMessage = 'An unknown error occurred';
    let statusCode = 500;
    
    if (error.response) {
      // The request was made and the server responded with an error status
      errorMessage = (error.response.data as any)?.detail || 
                    (error.response.data as any)?.message || 
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
