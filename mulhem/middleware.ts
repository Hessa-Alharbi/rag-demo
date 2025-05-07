import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

// This function can be marked `async` if using `await` inside
export function middleware(request: NextRequest) {
  const token = request.cookies.get('access_token')?.value
  const isAuthRoute = request.nextUrl.pathname.startsWith('/auth/')
  const isApiRoute = request.nextUrl.pathname.startsWith('/api/')
  
  // Ignore API routes
  if (isApiRoute) {
    return NextResponse.next()
  }

  // If no token and not on auth route, redirect to login
  if (!token && !isAuthRoute) {
    // Create the return URL with the original path
    const url = request.nextUrl.clone()
    const returnTo = encodeURIComponent(request.nextUrl.pathname + request.nextUrl.search)
    url.pathname = '/auth/login'
    url.search = `returnTo=${returnTo}`
    
    return NextResponse.redirect(url)
  }

  // If has token and on auth route, redirect to home
  if (token && isAuthRoute) {
    return NextResponse.redirect(new URL('/', request.url))
  }

  return NextResponse.next()
}

// See "Matching Paths" below to learn more
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
} 