"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/lib/auth-context"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import ProtectedRoute from "@/components/protected-route"

export default function ProfilePage() {
  const { user, isLoading, isAuthenticated } = useAuth()
  const router = useRouter()

  // Function to get user initials for the avatar
  const getUserInitials = () => {
    if (!user) return "U"
    
    if (user.first_name && user.last_name) {
      return `${user.first_name[0]}${user.last_name[0]}`
    }
    
    // Add a null check for username
    if (user.username) {
      return user.username.substring(0, 2).toUpperCase()
    }
    
    // Fallback if username is also undefined
    return "UN"
  }

  // Format date to be more readable
  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    })
  }

  return (
    <ProtectedRoute>
      <div className="container max-w-4xl py-10">
        <h1 className="text-3xl font-bold mb-6">Your Profile</h1>
        
        <div className="grid gap-6 md:grid-cols-[1fr_3fr]">
          {/* Left sidebar with avatar */}
          <div className="flex flex-col items-center space-y-4">
            <Avatar className="w-32 h-32 text-3xl">
              <AvatarFallback>{getUserInitials()}</AvatarFallback>
            </Avatar>
            <Button variant="outline" className="w-full">
              Change Avatar
            </Button>
          </div>
          
          {/* Right side with user details */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Account Information</CardTitle>
                <CardDescription>Your personal account details</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Username</p>
                    <p className="font-medium">{user?.username}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Email</p>
                    <p className="font-medium">{user?.email}</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">First Name</p>
                    <p className="font-medium">{user?.first_name || 'Not set'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Last Name</p>
                    <p className="font-medium">{user?.last_name || 'Not set'}</p>
                  </div>
                </div>
                
                <div>
                  <p className="text-sm text-muted-foreground">Account Created</p>
                  <p className="font-medium">{user?.created_at ? formatDate(user.created_at) : 'Unknown'}</p>
                </div>
              </CardContent>
              <CardFooter>
                <Button variant="outline" className="w-full">
                  Edit Profile
                </Button>
              </CardFooter>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Security</CardTitle>
                <CardDescription>Manage your account security</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="text-sm text-muted-foreground">Password</p>
                  <p className="font-medium">********</p>
                </div>
              </CardContent>
              <CardFooter>
                <Button variant="outline" className="w-full">
                  Change Password
                </Button>
              </CardFooter>
            </Card>
          </div>
        </div>
      </div>
    </ProtectedRoute>
  )
}

