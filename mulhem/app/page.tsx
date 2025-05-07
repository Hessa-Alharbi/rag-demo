"use client"

import { useEffect } from "react"
import Link from "next/link"
import Image from "next/image"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { useAuth } from "@/lib/auth-context"

export default function Home() {
  const { isAuthenticated, user, isLoading } = useAuth()
  const router = useRouter()

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push('/auth/login')
    }
  }, [isLoading, isAuthenticated, router])

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
      </div>
    )
  }

  // Non-authenticated users should be redirected, but show loading in case
  if (!isAuthenticated) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
      </div>
    )
  }

  // Authenticated user view
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <div className="flex flex-col items-center mb-8">
        <Image 
          src="/Transparent-Navid-Logo.png" 
          alt="Navid Logo" 
          width={120} 
          height={120} 
          className="mb-4"
        />
        <h1 className="text-4xl font-bold font-sans">Welcome, {user?.first_name || user?.username}</h1>

        {/* <p className="text-muted-foreground mt-2">What would you like to do today?</p> */}
      </div>
      <div className="space-y-4 w-full max-w-md">
        <Button asChild className="w-full">
          <Link href="/chat/new">Start New Chat</Link>
        </Button>
        <Button asChild className="w-full" variant="outline">
          <Link href="/chats">View Chat History</Link>
        </Button>
        <Button asChild variant="outline" className="w-full">
          <Link href="/settings">Settings</Link>
        </Button>
        <Button asChild variant="outline" className="w-full">
          <Link href="/profile">Profile</Link>
        </Button>
      </div>
    </div>
  )
}

