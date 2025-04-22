"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Search, MessageSquare, Trash2, Plus } from "lucide-react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"

interface Chat {
  id: string
  title: string
  lastMessage: string
  date: string
}

export default function Chats() {
  const [chats, setChats] = useState<Chat[]>([
    { id: "1", title: "Project Ideas", lastMessage: "What about a new app for...", date: "2023-06-01" },
    { id: "2", title: "Code Review", lastMessage: "The function looks good, but...", date: "2023-06-02" },
    { id: "3", title: "Bug Discussion", lastMessage: "I think I found the issue...", date: "2023-06-03" },
  ])

  const [searchTerm, setSearchTerm] = useState("")

  const filteredChats = chats.filter(
    (chat) =>
      chat.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      chat.lastMessage.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  const deleteChat = (id: string) => {
    setChats(chats.filter((chat) => chat.id !== id))
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">All Chats</h1>
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
          placeholder="Search chats"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-10"
        />
      </div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {filteredChats.map((chat) => (
          <Card key={chat.id}>
            <CardHeader>
              <CardTitle>{chat.title}</CardTitle>
              <CardDescription>{chat.date}</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">{chat.lastMessage}</p>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button asChild variant="outline">
                <Link href={`/chat/${chat.id}`}>
                  <MessageSquare className="mr-2 h-4 w-4" />
                  Open Chat
                </Link>
              </Button>
              <Button variant="destructive" size="icon" onClick={() => deleteChat(chat.id)}>
                <Trash2 className="h-4 w-4" />
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  )
}

