"use client"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Search, Plus, MessageSquare, Settings, User } from "lucide-react"
import { useSidebar } from "./sidebar-provider"

export default function Sidebar() {
  const { isOpen } = useSidebar()

  return (
    <div
      className={`fixed inset-y-0 left-0 z-50 w-64 bg-background transform transition-transform duration-200 ease-in-out border-r ${
        isOpen ? "translate-x-0" : "-translate-x-full"
      } lg:relative lg:translate-x-0`}
    >
      <div className="flex flex-col h-full">
        <div className="p-4">
          <div className="relative">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input placeholder="Search" className="pl-8" />
          </div>
        </div>
        <ScrollArea className="flex-1 px-3">
          <div className="space-y-4 py-4">
            <div className="px-3 py-2">
              <h2 className="mb-2 px-4 text-lg font-semibold tracking-tight">Menu</h2>
              <div className="space-y-1">
                <Button variant="secondary" className="w-full justify-start" asChild>
                  <Link href="/chats">
                    <MessageSquare className="mr-2 h-4 w-4" />
                    Chats
                  </Link>
                </Button>
                <Button variant="ghost" className="w-full justify-start" asChild>
                  <Link href="/settings">
                    <Settings className="mr-2 h-4 w-4" />
                    Settings
                  </Link>
                </Button>
                <Button variant="ghost" className="w-full justify-start" asChild>
                  <Link href="/profile">
                    <User className="mr-2 h-4 w-4" />
                    Profile
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </ScrollArea>
        <div className="p-4">
          <Button className="w-full" asChild>
            <Link href="/chat/new">
              <Plus className="mr-2 h-4 w-4" /> New Chat
            </Link>
          </Button>
        </div>
      </div>
    </div>
  )
}

