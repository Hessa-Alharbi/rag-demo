import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageSquare, Clock, Trash2, TrashIcon } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { useRecentConversations } from "@/hooks/useRecentConversations";
import { formatDistanceToNow } from 'date-fns';
import { ar } from 'date-fns/locale';
import { deleteConversation } from "@/lib/chat-service";
import { toast } from "@/components/ui/use-toast";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

export function RecentConversations() {
  const router = useRouter();
  const { conversations, isLoading, error, refetch } = useRecentConversations();
  const [showClearDialog, setShowClearDialog] = useState(false);

  // Format date to a relative time (e.g., "2 days ago")
  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return formatDistanceToNow(date, { addSuffix: true, locale: ar });
    } catch (e) {
      return dateString;
    }
  };

  // Navigate to the selected conversation
  const goToConversation = (id: string) => {
    router.push(`/chat/${id}`);
  };

  // Handle conversation deletion
  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation(); // Prevent navigation
    
    try {
      const success = await deleteConversation(id);
      if (success) {
        toast({
          title: "تم الحذف",
          description: "تم حذف المحادثة بنجاح",
        });
        // تحديث القائمة بعد الحذف
        refetch();
      } else {
        toast({
          title: "خطأ",
          description: "فشل في حذف المحادثة، حاول مرة أخرى",
          variant: "destructive",
        });
      }
    } catch (err) {
      console.error("Failed to delete conversation:", err);
      toast({
        title: "خطأ",
        description: "حدث خطأ أثناء حذف المحادثة",
        variant: "destructive",
      });
    }
  };

  // Handle clearing all conversations
  const handleClearAll = async () => {
    try {
      let successCount = 0;
      let failCount = 0;
      
      // نظراً لعدم وجود API لحذف جميع المحادثات دفعة واحدة، سنحذفها واحدة تلو الأخرى
      for (const conversation of conversations) {
        const success = await deleteConversation(conversation.id);
        if (success) {
          successCount++;
        } else {
          failCount++;
        }
      }
      
      if (successCount > 0) {
        toast({
          title: "تم مسح السجل",
          description: `تم حذف ${successCount} محادثة بنجاح${failCount > 0 ? ` (فشل حذف ${failCount})` : ''}`,
        });
        
        // مسح المحادثات من التخزين المحلي أيضاً
        localStorage.removeItem('recent_conversations');
        
        // تحديث القائمة
        refetch();
      } else if (failCount > 0) {
        toast({
          title: "فشل في المسح",
          description: "لم يتم حذف أي محادثة، يرجى المحاولة مرة أخرى",
          variant: "destructive",
        });
      }
    } catch (err) {
      console.error("Failed to clear conversations:", err);
      toast({
        title: "خطأ",
        description: "حدث خطأ أثناء مسح المحادثات",
        variant: "destructive",
      });
    } finally {
      setShowClearDialog(false);
    }
  };

  // Loading states
  if (isLoading) {
    return (
      <div className="space-y-2">
        <div className="flex justify-between items-center px-4 py-2">
          <h3 className="text-sm font-medium">المحادثات الأخيرة</h3>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <Clock className="h-4 w-4" />
          </Button>
        </div>
        {[1, 2, 3].map((i) => (
          <div key={i} className="px-2">
            <div className="flex items-center space-x-2 rtl:space-x-reverse p-2 rounded-md">
              <Skeleton className="h-8 w-8 rounded-full" />
              <div className="space-y-1 flex-1">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-3 w-2/3" />
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="px-4 py-2 text-sm text-red-500">
        {error}
        <Button variant="link" className="p-0 h-auto text-xs" onClick={() => refetch()}>
          إعادة المحاولة
        </Button>
      </div>
    );
  }

  // No conversations state
  if (conversations.length === 0) {
    return (
      <div className="px-4 py-2 text-sm text-center text-muted-foreground">
        لا توجد محادثات سابقة
      </div>
    );
  }

  // Render conversation list
  return (
    <>
      <div className="space-y-1">
        <div className="flex justify-between items-center px-4 py-2">
          <h3 className="text-sm font-medium">المحادثات الأخيرة</h3>
          <div className="flex gap-1">
            {conversations.length > 0 && (
              <Button 
                variant="ghost" 
                size="sm" 
                className="text-destructive hover:text-destructive hover:bg-destructive/10"
                onClick={() => setShowClearDialog(true)}
                title="مسح السجل"
              >
                <TrashIcon className="h-4 w-4 mr-1" />
                <span className="text-xs">مسح السجل</span>
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={() => refetch()} title="تحديث">
              <Clock className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <ScrollArea className="max-h-[300px]">
          {conversations.map((conversation) => (
            <div key={conversation.id} className="flex justify-between items-center gap-2 px-3 py-2 hover:bg-muted/40 rounded-md mx-1 my-1">
              <div className="flex-1 overflow-hidden cursor-pointer" onClick={() => goToConversation(conversation.id)}>
                <div className="flex items-start space-x-2 rtl:space-x-reverse">
                  <MessageSquare className="h-4 w-4 mt-1 flex-shrink-0 text-muted-foreground" />
                  <div className="flex flex-col items-start overflow-hidden">
                    <span className="text-sm font-medium truncate w-full text-start">
                      {conversation.title || 'محادثة بدون عنوان'}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {formatDate(conversation.updated_at)}
                    </span>
                    {conversation.last_message && (
                      <span className="text-xs text-muted-foreground truncate w-full text-start">
                        {conversation.last_message}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 flex-shrink-0 bg-destructive/10 hover:bg-destructive/30 rounded-full shadow-sm"
                onClick={(e) => handleDelete(e, conversation.id)}
                title="حذف المحادثة"
              >
                <Trash2 className="h-4 w-4 text-destructive" />
              </Button>
            </div>
          ))}
        </ScrollArea>
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
              onClick={handleClearAll}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              نعم، مسح السجل
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
} 