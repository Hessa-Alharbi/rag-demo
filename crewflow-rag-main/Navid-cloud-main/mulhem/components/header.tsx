"use client";

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { MoonIcon, SunIcon, Settings, User, Menu, MessageSquare } from "lucide-react"
import { useTheme } from "next-themes"
import { useSidebar } from "./sidebar-provider"

export default function Header() {
  const { setTheme } = useTheme()
  const { toggle } = useSidebar()

  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center">
        <div className="mr-4 hidden md:flex">
          <Link href="/" className="mr-6 flex items-center space-x-2">
            <MessageSquare className="h-6 w-6" />
            <span className="hidden font-bold sm:inline-block">Mulhem</span>
          </Link>
          <nav className="flex items-center space-x-6 text-sm font-medium">
            <Link href="/chats">Chats</Link>
            <Link href="/settings">Settings</Link>
            <Link href="/profile">Profile</Link>
          </nav>
        </div>
        <Button variant="ghost" size="icon" className="md:hidden" onClick={toggle}>
          <Menu className="h-5 w-5" />
        </Button>
        <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <div className="w-full flex-1 md:w-auto md:flex-none">
            <Button variant="outline" className="w-full md:w-auto">
              <MessageSquare className="mr-2 h-4 w-4" />
              New Chat
            </Button>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                <SunIcon className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                <MoonIcon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
                <span className="sr-only">Toggle theme</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setTheme("light")}>Light</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setTheme("dark")}>Dark</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setTheme("system")}>System</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  )
}

