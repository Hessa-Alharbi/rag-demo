import { NextRequest, NextResponse } from 'next/server'
import axios from 'axios'

// Get base URL for API server
const getBaseUrl = () => 'https://4161-2a02-cb80-4271-93aa-dc2c-9eea-2d8e-7325.ngrok-free.app';

export async function POST(
  request: NextRequest,
  { params }: { params: { nextauth: string[] } }
) {
  const authType = params.nextauth[0]
  const requestData = await request.json()

  try {
    switch (authType) {
      case 'login': {
        // Transform the data to match the OAuth2 form data format expected by the backend
        const formData = new URLSearchParams()
        formData.append('username', requestData.username)
        formData.append('password', requestData.password)
        
        const response = await axios.post(`${getBaseUrl()}/auth/login`, formData, {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
        })
        
        return NextResponse.json(response.data)
      }
      
      case 'register': {
        const response = await axios.post(`${getBaseUrl()}/auth/register`, requestData)
        return NextResponse.json(response.data)
      }
      
      case 'refresh': {
        const response = await axios.post(`${getBaseUrl()}/auth/refresh`, {
          refresh_token: requestData.refresh_token
        })
        return NextResponse.json(response.data)
      }
      
      case 'me': {
        const token = request.headers.get('Authorization')
        
        if (!token) {
          return NextResponse.json(
            { error: 'Authorization header is required' },
            { status: 401 }
          )
        }
        
        const response = await axios.get(`${getBaseUrl()}/auth/me`, {
          headers: {
            'Authorization': token
          }
        })
        
        return NextResponse.json(response.data)
      }
      
      default:
        return NextResponse.json(
          { error: 'Invalid auth endpoint' },
          { status: 400 }
        )
    }
  } catch (error: any) {
    console.error(`Auth API error (${authType}):`, error)
    
    return NextResponse.json(
      { 
        error: error.response?.data?.detail || 'An error occurred during authentication',
        status: error.response?.status || 500
      },
      { status: error.response?.status || 500 }
    )
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: { nextauth: string[] } }
) {
  const authType = params.nextauth[0]
  
  try {
    if (authType === 'me') {
      const token = request.headers.get('Authorization')
      
      if (!token) {
        return NextResponse.json(
          { error: 'Authorization header is required' },
          { status: 401 }
        )
      }
      
      const response = await axios.get(`${getBaseUrl()}/auth/me`, {
        headers: {
          'Authorization': token
        }
      })
      
      return NextResponse.json(response.data)
    } else {
      return NextResponse.json(
        { error: 'Invalid auth endpoint for GET request' },
        { status: 400 }
      )
    }
  } catch (error: any) {
    console.error(`Auth API error (GET ${authType}):`, error)
    
    return NextResponse.json(
      { 
        error: error.response?.data?.detail || 'An error occurred during authentication',
        status: error.response?.status || 500
      },
      { status: error.response?.status || 500 }
    )
  }
} 