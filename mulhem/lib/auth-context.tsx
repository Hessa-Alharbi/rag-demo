"use client"

import React, { createContext, useContext, useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import axios from "axios"
import apiClient from '@/lib/api-client'
import { getBaseUrl } from '@/lib/utils'

// Define the User interface based on the backend response
interface User {
  id: string
  username: string
  email: string
  first_name?: string
  last_name?: string
  is_active: boolean
  created_at: string
}

interface AuthContextType {
  user: User | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (username: string, password: string) => Promise<void>
  logout: () => void
  getUserData: () => Promise<void>
}

// Create the context
const AuthContext = createContext<AuthContextType | null>(null)

// Helper functions to manage cookies and localStorage
const setAuthTokens = (accessToken: string, refreshToken: string) => {
  // تأكد من أننا في المتصفح قبل محاولة استخدام localStorage
  if (typeof window !== 'undefined') {
    // Set in localStorage for client-side access
    localStorage.setItem('access_token', accessToken)
    localStorage.setItem('refresh_token', refreshToken)
    
    // Set in cookies for middleware access
    document.cookie = `access_token=${accessToken}; path=/; max-age=${30 * 60}; SameSite=Strict` // 30 minutes
    document.cookie = `refresh_token=${refreshToken}; path=/; max-age=${7 * 24 * 60 * 60}; SameSite=Strict` // 7 days
  }
}

const clearAuthTokens = () => {
  // تأكد من أننا في المتصفح قبل محاولة استخدام localStorage
  if (typeof window !== 'undefined') {
    // Clear from localStorage
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    
    // Clear from cookies
    document.cookie = 'access_token=; path=/; max-age=0; SameSite=Strict'
    document.cookie = 'refresh_token=; path=/; max-age=0; SameSite=Strict'
  }
}

// Auth provider component
export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const router = useRouter()

  // Set up axios instance with authentication headers
  const api = axios.create({
    baseURL: getBaseUrl(),
  })

  // Add request interceptor to include token in requests
  api.interceptors.request.use((config) => {
    // الحصول على التوكن من التخزين المحلي - فقط عندما يكون المستعرض متاحًا
    let token;
    if (typeof window !== 'undefined') {
      token = localStorage.getItem("access_token")
    }
    
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  })

  // Add response interceptor to handle token refresh
  api.interceptors.response.use(
    (response) => response,
    async (error) => {
      const originalRequest = error.config
      
      // If the error is 401 and we haven't tried refreshing the token yet
      if (error.response?.status === 401 && !originalRequest._retry) {
        originalRequest._retry = true
        
        try {
          // تأكد من وجود المتصفح
          let refreshToken = null;
          if (typeof window !== 'undefined') {
            refreshToken = localStorage.getItem("refresh_token")
          }
          
          if (!refreshToken) {
            throw new Error("No refresh token")
          }
          
          // Try to refresh the token using our API client
          const response = await axios.post(
            `${getBaseUrl()}/auth/refresh`, 
            { refresh_token: refreshToken }
          )
          
          if (response.data.access_token) {
            // Update stored tokens
            setAuthTokens(response.data.access_token, response.data.refresh_token)
            
            // Update the original request with the new token
            originalRequest.headers.Authorization = `Bearer ${response.data.access_token}`
            return axios(originalRequest)
          }
        } catch (refreshError) {
          // If refresh fails, log out the user
          logout()
          return Promise.reject(refreshError)
        }
      }
      
      return Promise.reject(error)
    }
  )

  const getUserData = async () => {
    try {
      if (typeof window === 'undefined') {
        setIsLoading(false)
        return null
      }
      
      const token = localStorage.getItem("access_token")
      if (!token) {
        console.log('No token found, user is not authenticated')
        setUser(null)
        setIsLoading(false)
        return
      }
      
      console.log('Fetching user data with token:', token.substring(0, 15) + '...')
      
      // إضافة محاولات متعددة مع تأخير
      let attempts = 0;
      const maxAttempts = 3;
      
      const fetchWithRetry = async () => {
        try {
          // استخدام مفتاح عشوائي لمنع التخزين المؤقت
          const timestamp = new Date().getTime();
          
          // سجل طريقة الطلب مع تفاصيل الرؤوس
          console.log('Making auth/me request with headers:', {
            'Authorization': `Bearer ${token.substring(0, 15)}...`,
            'Timestamp': timestamp
          });
          
          // استخدام fetch مباشرة بدلاً من apiClient
          const response = await fetch(`${getBaseUrl()}/auth/me?_t=${timestamp}`, {
            method: 'GET',
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
              'Cache-Control': 'no-cache, no-store',
              'Pragma': 'no-cache'
            },
            credentials: 'omit' // لا ترسل الكوكيز
          });
          
          if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
          }
          
          const userData = await response.json();
          console.log('User data received successfully:', userData);
          setUser(userData);
          console.log('User state updated, authenticated:', !!userData);
          return userData;
        } catch (error) {
          console.error(`Attempt ${attempts + 1}/${maxAttempts} failed:`, error);
          
          if (attempts < maxAttempts - 1) {
            attempts++;
            const delay = 500 * attempts; // تأخير متزايد
            console.log(`Retrying in ${delay}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
            return fetchWithRetry(); // إعادة المحاولة
          }
          
          // إعادة إلقاء الخطأ إذا استنفدنا جميع المحاولات
          throw error;
        }
      };
      
      return await fetchWithRetry();
    } catch (error) {
      console.error("Error fetching user data:", error)
      // If there's an authentication error, clear the tokens
      clearAuthTokens()
      setUser(null)
    } finally {
      setIsLoading(false)
    }
    return null
  }

  // Login function
  const login = async (username: string, password: string) => {
    setIsLoading(true)
    
    try {
      console.log('Attempting login for user:', username)
      
      // Use a similar strategy to the file upload strategy
      let retryCount = 0;
      const maxRetries = 3;
      const initialDelay = 2000; // Longer delay to allow server to recover
      
      // Local function for retrying with backoff
      const attemptLogin = async () => {
        try {
          // Use URLSearchParams for form data
          const formData = new URLSearchParams()
          formData.append('username', username)
          formData.append('password', password)

          // Use a delay before sending the request
          await new Promise(resolve => setTimeout(resolve, 500));
          
          // Get the API URL
          const apiUrl = getBaseUrl();
          console.log('Using API URL:', apiUrl);
          
          // Try using direct fetch instead of axios with additional options
          const response = await fetch(`${apiUrl}/auth/login`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded',
              'Accept': 'application/json',
              'Origin': window.location.origin,
              'Access-Control-Request-Method': 'POST',
            },
            body: formData.toString(),
            // Other settings
            credentials: 'omit', // Avoid sending cookies
            cache: 'no-cache',
            redirect: 'follow',
            mode: 'cors', // Explicitly request CORS mode
            // Add longer timeout
            signal: AbortSignal.timeout(15000), // 15 second timeout
          });
          
          if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
          }
          
          const data = await response.json();
          return data;
        } catch (error: any) {
          console.error("Login attempt failed:", error);
          
          // Check if it's a network error (Failed to fetch)
          if (error.message && (error.message.includes('Failed to fetch') || error.message.includes('NetworkError'))) {
            console.error("Network error - connection issue to server");
            
            // If we have retries left, try again
            if (retryCount < maxRetries) {
              retryCount++;
              const backoffDelay = initialDelay * Math.pow(2, retryCount - 1);
              console.log(`Network error - retrying login after ${backoffDelay}ms delay (attempt ${retryCount} of ${maxRetries})`);
              await new Promise(resolve => setTimeout(resolve, backoffDelay));
              return attemptLogin();
            }
            
            // Provide more specific error message
            throw new Error("Failed to connect to the server. Please check your internet connection and make sure the server is running.");
          }
          
          // Check if it's a server error (500)
          if (error.message && error.message.includes('500')) {
            console.error("Server returned 500 error - internal server error");
            throw error; // Don't retry on 500 errors
          }
          
          // Retry logic for 503 errors or network issues
          if ((error.message && (error.message.includes('503') || error.message.includes('network'))) && retryCount < maxRetries) {
            retryCount++;
            const backoffDelay = initialDelay * Math.pow(2, retryCount - 1);
            console.log(`Retrying login after ${backoffDelay}ms delay (attempt ${retryCount} of ${maxRetries})`);
            await new Promise(resolve => setTimeout(resolve, backoffDelay));
            return attemptLogin();
          }
          
          throw error;
        }
      };
      
      // Call the login function with retry logic
      const tokens = await attemptLogin();
      
      // Handle the response
      if (tokens && tokens.access_token) {
        console.log('Login successful, storing tokens')
        setAuthTokens(tokens.access_token, tokens.refresh_token)
        
        // Wait a short time to ensure tokens are saved
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Fetch user data after successful login
        console.log('Now fetching user data with the new token');
        const userData = await getUserData()
        
        if (userData) {
          console.log('Login complete, redirecting to dashboard')
          // Check if we have a returnTo URL in the query string
          const params = new URLSearchParams(window.location.search)
          const returnTo = params.get('returnTo')
          
          // تحديث حالة المستخدم أولاً
          setUser(userData);
          setIsLoading(false);
          
          // استخدام عنوان مطلق بدلاً من عنوان نسبي
          const targetUrl = returnTo || '/';
          console.log('Redirecting to:', targetUrl);
          
          // تنفيذ التوجيه مباشرة بدون تأخير
          try {
            // استخدام واجهة router.push مع انتظار العملية
            router.push(targetUrl);
            
            // إضافة إجراء احتياطي في حالة فشل التوجيه الأول
            setTimeout(() => {
              console.log('Checking if redirect was successful...');
              const currentPath = window.location.pathname;
              if (currentPath.includes('login')) {
                console.log('Redirect may have failed, trying alternate method');
                window.location.href = targetUrl;
              }
            }, 1000);
          } catch (routerError) {
            console.error('Error during navigation:', routerError);
            // استخدام window.location كحل بديل
            console.log('Falling back to window.location.href redirect');
            window.location.href = targetUrl;
          }
        } else {
          throw new Error('Failed to get user data after login')
        }
      } else {
        throw new Error('Login response did not contain tokens')
      }
    } catch (error) {
      console.error("Login error:", error)
      // Clear any partial auth data
      clearAuthTokens()
      setUser(null)
      setIsLoading(false)
      throw error
    }
  }

  // Logout function
  const logout = () => {
    console.log('Logging out user')
    clearAuthTokens()
    setUser(null)
    
    // تحسين آلية التوجيه
    console.log('Redirecting to login page')
    
    try {
      // استخدام واجهة router.push أولاً
      router.push("/auth/login")
      
      // إضافة إجراء احتياطي في حالة فشل التوجيه الأول
      setTimeout(() => {
        const currentPath = window.location.pathname;
        if (!currentPath.includes('login')) {
          console.log('Redirect may have failed, trying alternate method');
          window.location.href = "/auth/login";
        }
      }, 500);
    } catch (routerError) {
      console.error('Error during navigation:', routerError);
      // استخدام window.location كحل بديل
      window.location.href = "/auth/login";
    }
  }

  // Check if user is logged in on initial load
  useEffect(() => {
    const checkAuth = async () => {
      console.log('Checking authentication on initial load')
      try {
        // تأكد من أننا في بيئة المتصفح
        if (typeof window === 'undefined') {
          console.log('Not in browser environment, skipping auth check');
          setIsLoading(false);
          return;
        }
        
        const token = localStorage.getItem('access_token');
        if (!token) {
          console.log('No token found, user is not authenticated');
          setUser(null);
          setIsLoading(false);
          return;
        }
        
        // تأخير بسيط للسماح بتهيئة المتصفح بشكل كامل
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // استدعاء وظيفة الحصول على بيانات المستخدم
        await getUserData();
      } catch (error) {
        console.error('Error during initial auth check:', error);
        setUser(null);
        setIsLoading(false);
      }
    }
    
    checkAuth()
  }, [])

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        logout,
        getUserData,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

// Custom hook to use the auth context
export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}

// Export a function to check authentication on the server side
export async function getServerSideAuth(cookies: any) {
  try {
    const token = cookies.get("access_token")?.value
    
    if (!token) {
      return { isAuthenticated: false, user: null }
    }
    
    // استخدام المسار المباشر للواجهة الخلفية
    const apiUrl = process.env.NODE_ENV === 'production' 
      ? '/api/auth/me' 
      : 'http://localhost:8000/api/auth/me'
      
    const response = await axios.get(apiUrl, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    
    return {
      isAuthenticated: true,
      user: response.data,
    }
  } catch (error) {
    return { isAuthenticated: false, user: null }
  }
} 
