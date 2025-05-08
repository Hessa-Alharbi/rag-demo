import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from 'axios';
import { getBaseUrl } from './utils';
import axiosRetry from 'axios-retry';

interface ExtendedAxiosRequestConfig extends AxiosRequestConfig {
  retryCount?: number;
  maxRetries?: number;
  _retry?: boolean;
}

// تكوين عميل axios مع إعدادات محسنة
const apiClient = axios.create({
  baseURL: getBaseUrl(),
  timeout: 60000,  // زيادة المهلة إلى 60 ثانية
  headers: {
    'Content-Type': 'application/json',
  },
});

// سجل عنوان URL الأساسي عند إنشاء العميل
console.log('API client created with baseURL:', apiClient.defaults.baseURL);
console.log('Frontend application running at: https://rag-demo-frontend-8eyx.onrender.com');
console.log('IMPORTANT: Please add this frontend domain to the CORS allowed origins in your backend settings');

// إضافة معالج للطلبات لإضافة التوكن إذا كان متاحاً
apiClient.interceptors.request.use(
  (config) => {
    // إضافة رؤوس إضافية لتجنب مشاكل CORS
    config.headers['Access-Control-Allow-Origin'] = '*';
    
    // حصول على التوكن من التخزين المحلي إذا كان متاحاً
    const accessToken = localStorage.getItem('accessToken');
    
    if (accessToken) {
      try {
        // التحقق من صلاحية التوكن (باستخدام تحقق بسيط)
        const base64Url = accessToken.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(c => {
          return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));

        const decodedToken = JSON.parse(jsonPayload);
        const currentTime = Date.now() / 1000;
        
        if (decodedToken.exp && decodedToken.exp > currentTime) {
          // التوكن صالح
          config.headers.Authorization = `Bearer ${accessToken}`;
        } else {
          // التوكن منتهي الصلاحية - لا تُستخدم
          console.warn('Token expired, not using it');
          localStorage.removeItem('accessToken');
        }
      } catch (error) {
        console.error('Error decoding token:', error);
        localStorage.removeItem('accessToken');
      }
    }
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// تنفيذ وظيفتنا الخاصة لإعادة المحاولة لتجنب مشكلة التوافق في المكتبة
const setupCustomRetry = (instance: AxiosInstance) => {
  // تحديد عدد المحاولات والتأخير
  const retries = 3;
  
  // إضافة معترض للاستجابة لمعالجة الأخطاء وإعادة المحاولة
  instance.interceptors.response.use(undefined, async (error) => {
    const config = error.config;
    
    // إذا لم تكن هناك إعدادات أو تم تجاوز عدد المحاولات
    if (!config || !error.response || config.__retryCount >= retries) {
      return Promise.reject(error);
    }
    
    // التحقق مما إذا كان الخطأ يستحق إعادة المحاولة
    const shouldRetry = error.response.status >= 500 || !error.response;
    if (!shouldRetry) {
      return Promise.reject(error);
    }
    
    // زيادة عداد المحاولات
    config.__retryCount = (config.__retryCount || 0) + 1;
    
    // حساب فترة الانتظار بشكل تصاعدي
    const delay = config.__retryCount * 3000;
    console.log(`Retrying request (${config.__retryCount}/${retries}) after ${delay}ms - ${error.message}`);
    
    // انتظار قبل إعادة المحاولة
    await new Promise(resolve => setTimeout(resolve, delay));
    
    // إعادة محاولة الطلب
    return instance(config);
  });
  
  return instance;
};

// تكوين آلية إعادة المحاولة المخصصة
setupCustomRetry(apiClient);

// إضافة معالج للاستجابات للتعامل مع الأخطاء بشكل أفضل
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error: AxiosError) => {
    if (error.response) {
      // تسجيل معلومات الخطأ للمساعدة في التشخيص
      console.error('API Error Response:', {
        status: error.response.status,
        data: error.response.data,
        url: error.config?.url,
        method: error.config?.method
      });
      
      // التعامل مع أخطاء محددة
      if (error.response.status === 401) {
        // غير مصرح به - تنظيف التوكن
        localStorage.removeItem('accessToken');
        localStorage.removeItem('refreshToken');
        
        // يمكن إضافة منطق إعادة التوجيه إلى صفحة تسجيل الدخول هنا
      }
    } else if (error.request) {
      // تم إجراء الطلب لكن لم يتم استلام استجابة
      console.error('No response received:', error.request);
    } else {
      // حدث خطأ عند إعداد الطلب
      console.error('Request setup error:', error.message);
    }
    
    return Promise.reject(error);
  }
);

export default apiClient;
