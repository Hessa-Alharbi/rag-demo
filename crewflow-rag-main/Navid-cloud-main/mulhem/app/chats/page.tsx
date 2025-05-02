"use client"

import { useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Search, MessageSquare, Trash2, Plus, Loader2 } from "lucide-react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { useRecentConversations } from "@/hooks/useRecentConversations"
import { deleteConversation } from "@/lib/chat-service"
import { toast } from "@/components/ui/use-toast"
import { formatDistanceToNow } from "date-fns"

export default function Chats() {
  const router = useRouter()
  const { conversations, isLoading, error, refetch } = useRecentConversations()
  const [searchTerm, setSearchTerm] = useState("")

  const filteredConversations = conversations.filter(
    (conversation) =>
      conversation.title?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      conversation.last_message?.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const handleDelete = async (id: string) => {
    try {
      const success = await deleteConversation(id)
      if (success) {
        toast({
          title: "Deleted",
          description: "Conversation deleted successfully",
        })
        refetch()
      } else {
        toast({
          title: "Error",
          description: "Failed to delete conversation",
          variant: "destructive",
        })
      }
    } catch (err) {
      console.error("Failed to delete conversation:", err)
      toast({
        title: "Error",
        description: "An error occurred while deleting the conversation",
        variant: "destructive",
      })
    }
  }

  // Format date to a relative time (e.g., "2 days ago")
  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString)
      return formatDistanceToNow(date, { addSuffix: true })
    } catch (e) {
      return dateString
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">Chat History</h1>
        <Button asChild>
          <Link href="/chat/new">
            <Plus className="mr-2 h-4 w-4" />
            New Chat
          </Link>
        </Button>
      </div>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search conversations"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-10"
        />
      </div>

      {isLoading ? (
        <div className="flex justify-center items-center py-10">
          <Loader2 className="h-10 w-10 animate-spin text-muted-foreground" />
        </div>
      ) : error ? (
        <div className="text-center py-10 text-red-500">
          <p>{error}</p>
          <Button variant="outline" onClick={() => refetch()} className="mt-4">
            Try Again
          </Button>
        </div>
      ) : filteredConversations.length === 0 ? (
        <div className="text-center py-10 text-muted-foreground">
          {searchTerm ? "No conversations match your search" : "No conversation history yet"}
          <div className="mt-4">
            <Button asChild>
              <Link href="/chat/new">Start a new conversation</Link>
            </Button>
          </div>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredConversations.map((conversation) => (
            <Card key={conversation.id} className="flex flex-col">
              <CardHeader>
                <CardTitle className="truncate">{conversation.title || "Untitled conversation"}</CardTitle>
                <CardDescription>{formatDate(conversation.updated_at)}</CardDescription>
              </CardHeader>
              <CardContent className="flex-grow">
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {conversation.last_message || "No messages"}
                </p>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button asChild variant="outline">
                  <Link href={`/chat/${conversation.id}`}>
                    <MessageSquare className="mr-2 h-4 w-4" />
                    Open Chat
                  </Link>
                </Button>
                <Button 
                  variant="destructive" 
                  size="icon" 
                  onClick={() => handleDelete(conversation.id)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}

