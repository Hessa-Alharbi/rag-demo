"use client"

import { useEffect, useRef } from 'react'
import { useAuth } from '@/lib/auth-context'
import { usePathname, useRouter } from 'next/navigation'

const publicPaths = ['/auth/login', '/auth/register', '/auth/forgot-password']

export default function AuthStateMonitor() {
  const { isAuthenticated, isLoading, getUserData } = useAuth()
  const pathname = usePathname()
  const router = useRouter()
  const isRedirectingRef = useRef(false)
  
  // تحقق من حالة المصادقة عند تغيير المسار
  useEffect(() => {
    const checkAuthState = async () => {
      // تجاهل المسارات العامة
      if (publicPaths.includes(pathname || '')) {
        isRedirectingRef.current = false
        return
      }
      
      // إذا كان المستخدم غير مصادق ولسنا في صفحة تسجيل الدخول، أعد التوجيه
      if (!isLoading && !isAuthenticated && !isRedirectingRef.current) {
        console.log('User not authenticated, redirecting to login page, current path:', pathname)
        isRedirectingRef.current = true
        router.push('/auth/login')
      } else if (isAuthenticated) {
        isRedirectingRef.current = false
      }
    }
    
    checkAuthState()
  }, [pathname, isAuthenticated, isLoading, router])
  
  // تحقق من صلاحية التوكن كل 5 دقائق
  useEffect(() => {
    const interval = setInterval(() => {
      if (typeof window !== 'undefined') {
        const token = localStorage.getItem('access_token')
        if (token) {
          getUserData()
        }
      }
    }, 5 * 60 * 1000) // 5 دقائق
    
    return () => clearInterval(interval)
  }, [getUserData])
  
  // لا حاجة لعرض أي شيء، هذا المكون للمنطق فقط
  return null
} 
