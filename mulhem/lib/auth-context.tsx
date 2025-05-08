"use client"

import React, { createContext, useContext, useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import axios from "axios"
import apiClient, { getBaseUrl } from '@/lib/api-client'

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
  // Set in localStorage for client-side access
  localStorage.setItem('access_token', accessToken)
  localStorage.setItem('refresh_token', refreshToken)
  
  // Set in cookies for middleware access
  document.cookie = `access_token=${accessToken}; path=/; max-age=${30 * 60}; SameSite=Strict` // 30 minutes
  document.cookie = `refresh_token=${refreshToken}; path=/; max-age=${7 * 24 * 60 * 60}; SameSite=Strict` // 7 days
}

const clearAuthTokens = () => {
  // Clear from localStorage
  localStorage.removeItem('access_token')
  localStorage.removeItem('refresh_token')
  
  // Clear from cookies
  document.cookie = 'access_token=; path=/; max-age=0; SameSite=Strict'
  document.cookie = 'refresh_token=; path=/; max-age=0; SameSite=Strict'
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
    const token = localStorage.getItem("access_token")
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
          const refreshToken = localStorage.getItem("refresh_token")
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
      const token = localStorage.getItem("access_token")
      if (!token) {
        console.log('No token found, user is not authenticated')
        setUser(null)
        setIsLoading(false)
        return
      }
      
      console.log('Fetching user data with token:', token.substring(0, 15) + '...')
      // Use our API client instead
      const response = await apiClient.get("/auth/me")
      
      if (response.data) {
        console.log('User data received successfully:', response.data)
        setUser(response.data)
        console.log('User state updated, authenticated:', !!response.data)
        return response.data
      }
    } catch (error) {
      console.error("Error fetching user data:", error)
      // If there's an authentication error, clear the tokens
      if (axios.isAxiosError(error) && error.response?.status === 401) {
        console.log('Authentication error, clearing tokens')
        clearAuthTokens()
      }
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
      
      // Use URLSearchParams for form data
      const formData = new URLSearchParams()
      formData.append('username', username)
      formData.append('password', password)
      
      // استخدام التأخير البسيط قبل إرسال الطلب
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Use direct axios call with proper content type for login
      const response = await axios.post(`${getBaseUrl()}/auth/login`, formData.toString(), {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        // إضافة خيارات إضافية لتجاوز مشاكل CORS المحتملة
        withCredentials: false
      })
      
      if (response.data.access_token) {
        console.log('Login successful, storing tokens')
        setAuthTokens(response.data.access_token, response.data.refresh_token)
        
        // Fetch user data after successful login
        const userData = await getUserData()
        
        if (userData) {
          console.log('Login complete, redirecting to dashboard')
          // Check if we have a returnTo URL in the query string
          const params = new URLSearchParams(window.location.search)
          const returnTo = params.get('returnTo')
          
          // إضافة تأخير قبل التوجيه لضمان تحديث حالة المستخدم في كل مكان
          setTimeout(() => {
            if (returnTo) {
              // Redirect to the original URL
              console.log('Redirecting to:', returnTo)
              router.push(returnTo)
            } else {
              // Redirect to home page as usual
              console.log('Redirecting to dashboard /')
              router.push('/')
            }
          }, 500); // تأخير 500 مللي ثانية
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
    router.push("/auth/login")
  }

  // Check if user is logged in on initial load
  useEffect(() => {
    const checkAuth = async () => {
      console.log('Checking authentication on initial load')
      await getUserData()
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
