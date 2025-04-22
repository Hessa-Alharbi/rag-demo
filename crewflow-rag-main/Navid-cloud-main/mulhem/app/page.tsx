import Link from "next/link"
import { Button } from "@/components/ui/button"

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-4xl font-bold mb-8">Welcome to Mulhem</h1>
      <div className="space-y-4">
        <Button asChild className="w-full">
          <Link href="/chats">View All Chats</Link>
        </Button>
        <Button asChild className="w-full">
          <Link href="/chat/new">Start New Chat</Link>
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

