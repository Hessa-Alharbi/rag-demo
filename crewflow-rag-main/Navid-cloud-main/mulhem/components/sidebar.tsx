"use client"
import Link from "next/link"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Search, Plus, MessageSquare, Settings, User, Trash2, LogOut } from "lucide-react"
import { useSidebar } from "./sidebar-provider"
import { RecentConversations } from "./recent-conversations"
import { Separator } from "./ui/separator"
import { clearAllConversations } from "@/lib/chat-service"
import { toast } from "@/components/ui/use-toast"
import { useAuth } from "@/lib/auth-context"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"

export default function Sidebar() {
  const { isOpen } = useSidebar()
  const { isAuthenticated, logout, user } = useAuth()
  const [showClearDialog, setShowClearDialog] = useState(false)
  const [isClearing, setIsClearing] = useState(false)

  const handleClearHistory = async () => {
    try {
      setIsClearing(true)
      const result = await clearAllConversations()
      
      if (result.success) {
        toast({
          title: "History Cleared",
          description: "All conversations have been successfully deleted",
        })
        
        // Clear local storage as well
        localStorage.removeItem('recent_conversations')
        
        // Update UI and redirect user to home page
        window.location.href = "/"
      } else {
        toast({
          title: "Error",
          description: "Failed to clear conversations",
          variant: "destructive",
        })
      }
    } catch (err) {
      console.error("Failed to clear history:", err)
      toast({
        title: "Error",
        description: "An error occurred while clearing conversations",
        variant: "destructive",
      })
    } finally {
      setIsClearing(false)
      setShowClearDialog(false)
    }
  }

  // If user is not authenticated, don't show the sidebar
  if (!isAuthenticated) {
    return null
  }

  return (
    <>
      <div
        className={`fixed inset-y-0 left-0 z-50 w-[400px] bg-background transform transition-transform duration-200 ease-in-out border-r ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        } lg:relative lg:translate-x-0`}
      >
        <div className="flex flex-col h-full">
          {/* User profile section */}
          <div className="p-4 border-b">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold">
                {user?.first_name?.[0] || user?.username?.[0] || "U"}
              </div>
              <div className="flex-1 overflow-hidden">
                <p className="font-medium truncate">
                  {user?.first_name && user?.last_name 
                    ? `${user.first_name} ${user.last_name}` 
                    : user?.username}
                </p>
                <p className="text-xs text-muted-foreground truncate">{user?.email}</p>
              </div>
            </div>
          </div>
          
          {/* Navigation */}
          <div className="p-4">
            <div className="space-y-1">
              <Button variant="ghost" className="w-full justify-start" asChild>
                <Link href="/">
                  <MessageSquare className="mr-2 h-4 w-4" />
                  Dashboard
                </Link>
              </Button>
              <Button variant="ghost" className="w-full justify-start" asChild>
                <Link href="/profile">
                  <User className="mr-2 h-4 w-4" />
                  Profile
                </Link>
              </Button>
              <Button variant="ghost" className="w-full justify-start" asChild>
                <Link href="/settings">
                  <Settings className="mr-2 h-4 w-4" />
                  Settings
                </Link>
              </Button>
            </div>
          </div>
          
          <Separator className="my-2" />
          
          <ScrollArea className="flex-1 px-3">
            <div className="space-y-4 py-4">
              {/* Recent Conversations Section */}
              <div className="px-3 py-2">
                <RecentConversations />
              </div>
            </div>
          </ScrollArea>
          <div className="p-4 space-y-2">
            <Button className="w-full" asChild>
              <Link href="/chat/new">
                <Plus className="mr-2 h-4 w-4" /> New Chat
              </Link>
            </Button>
            <Button 
              variant="outline" 
              className="w-full bg-destructive/5 hover:bg-destructive/10 text-destructive hover:text-destructive" 
              onClick={() => setShowClearDialog(true)}
              disabled={isClearing}
            >
              <Trash2 className="mr-2 h-4 w-4" /> Clear History
            </Button>
            <Button 
              variant="ghost" 
              className="w-full" 
              onClick={logout}
            >
              <LogOut className="mr-2 h-4 w-4" /> Logout
            </Button>
          </div>
        </div>
      </div>

      <AlertDialog open={showClearDialog} onOpenChange={setShowClearDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear Conversation History</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete all previous conversations? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={handleClearHistory}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={isClearing}
            >
              {isClearing ? "Clearing..." : "Yes, Clear History"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}

