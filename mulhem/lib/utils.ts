import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Define backend API URL
export function getBaseUrl() {
  // Use the environment variable if available
  if (process.env.NEXT_PUBLIC_API_URL) {
    console.log('Using API URL from environment:', process.env.NEXT_PUBLIC_API_URL);
    return process.env.NEXT_PUBLIC_API_URL;
  }
  
  // For browser environment, check if we're in development mode
  if (typeof window !== 'undefined') {
    // Check if running locally
    const hostname = window.location.hostname;
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      console.log('Using local development API URL');
      return 'http://localhost:8000';
    }
    
    // Try to use the same origin for production
    // Only if not localhost and not running on a port
    if (!window.location.port || window.location.port === '443' || window.location.port === '80') {
      const apiUrl = `${window.location.protocol}//${window.location.hostname}/api`;
      console.log('Using same-origin API URL:', apiUrl);
      return apiUrl;
    }
  }
  
  // Fallback to local development URL for Node.js environment
  if (process.env.NODE_ENV === 'development') {
    console.log('Using development fallback API URL');
    return 'http://localhost:8000';
  }
  
  // Production fallback - Use Railway address
  console.log('Using production fallback API URL');
  return 'https://ingenious-transformation-production-be7c.up.railway.app';
}
