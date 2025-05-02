"use client"

import { usePathname } from "next/navigation"
import Sidebar from "@/components/sidebar"
import { useAuth } from "@/lib/auth-context"

export default function ClientSidebar() {
  const pathname = usePathname()
  const { isAuthenticated } = useAuth()
  
  // Only show sidebar for authenticated users on chat pages
  const showSidebar = isAuthenticated && 
    pathname?.startsWith('/chat/') && 
    pathname !== '/chat/new'
  
  if (!showSidebar) {
    return null
  }
  
  return <Sidebar />
} 