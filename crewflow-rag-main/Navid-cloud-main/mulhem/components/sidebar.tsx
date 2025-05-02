"use client"
import Link from "next/link"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Search, Plus, MessageSquare, Settings, User, Trash2 } from "lucide-react"
import { useSidebar } from "./sidebar-provider"
import { RecentConversations } from "./recent-conversations"
import { Separator } from "./ui/separator"
import { clearAllConversations } from "@/lib/chat-service"
import { toast } from "@/components/ui/use-toast"
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
  const [showClearDialog, setShowClearDialog] = useState(false)
  const [isClearing, setIsClearing] = useState(false)

  const handleClearHistory = async () => {
    try {
      setIsClearing(true)
      const result = await clearAllConversations()
      
      if (result.success) {
        toast({
          title: "تم مسح السجل",
          description: "تم مسح جميع المحادثات بنجاح",
        })
        
        // مسح التخزين المحلي أيضاً
        localStorage.removeItem('recent_conversations')
        
        // تحديث الواجهة وإعادة توجيه المستخدم للصفحة الرئيسية
        window.location.href = "/"
      } else {
        toast({
          title: "خطأ",
          description: "فشل في مسح المحادثات",
          variant: "destructive",
        })
      }
    } catch (err) {
      console.error("Failed to clear history:", err)
      toast({
        title: "خطأ",
        description: "حدث خطأ أثناء مسح المحادثات",
        variant: "destructive",
      })
    } finally {
      setIsClearing(false)
      setShowClearDialog(false)
    }
  }

  return (
    <>
      <div
        className={`fixed inset-y-0 left-0 z-50 w-70 bg-background transform transition-transform duration-200 ease-in-out border-r ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        } lg:relative lg:translate-x-0`}
      >
        <div className="flex flex-col h-full">
          {/* <div className="p-4">
            <div className="relative">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input placeholder="Search" className="pl-8" />
            </div>
          </div> */}
          <ScrollArea className="flex-1 px-3">
            <div className="space-y-4 py-4">
              {/* <div className="px-3 py-2">
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
              </div> */}
              
              <Separator className="my-4" />
              
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
          </div>
        </div>
      </div>

      <AlertDialog open={showClearDialog} onOpenChange={setShowClearDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>مسح سجل المحادثات</AlertDialogTitle>
            <AlertDialogDescription>
              هل أنت متأكد أنك تريد حذف جميع المحادثات السابقة؟ هذا الإجراء لا يمكن التراجع عنه.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>إلغاء</AlertDialogCancel>
            <AlertDialogAction 
              onClick={handleClearHistory}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={isClearing}
            >
              {isClearing ? "جاري المسح..." : "نعم، مسح السجل"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}

