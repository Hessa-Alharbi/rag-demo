import "./globals.css"
import "@/styles/markdown.css"
import { Inter } from "next/font/google"
import { ThemeProvider } from "@/components/theme-provider"
import Header from "@/components/header"
import Sidebar from "@/components/sidebar"
import { SidebarProvider } from "@/components/sidebar-provider"
import type React from "react"
import ClientSidebar from "@/components/client-sidebar"
import { AuthProvider } from "@/lib/auth-context"
import AuthStateMonitor from "@/components/auth-state-monitor"

const inter = Inter({ subsets: ["latin"] })

export const metadata = {
  title: "Modern Navid",
  description: "A sophisticated, user-friendly Navid interface",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <AuthProvider>
          <AuthStateMonitor />
          <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
            <SidebarProvider>
              <div className="flex h-screen overflow-hidden">
                <ClientSidebar />
                <div className="flex flex-col flex-1 overflow-hidden">
                  <Header />
                  <main className="flex-1 overflow-auto p-6">{children}</main>
                </div>
              </div>
            </SidebarProvider>
          </ThemeProvider>
        </AuthProvider>
      </body>
    </html>
  )
}

