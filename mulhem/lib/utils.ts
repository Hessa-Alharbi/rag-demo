import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
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
  return 'https://ingenious-transformation-production-be7c.up.railway.app';
};
