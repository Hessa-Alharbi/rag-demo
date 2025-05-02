import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageSquare, Clock, Trash2, TrashIcon } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { useRecentConversations } from "@/hooks/useRecentConversations";
import { formatDistanceToNow } from 'date-fns';
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
      return formatDistanceToNow(date, { addSuffix: true });
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
          title: "Deleted",
          description: "Conversation deleted successfully",
        });
        // Update list after deletion
        refetch();
      } else {
        toast({
          title: "Error",
          description: "Failed to delete conversation, please try again",
          variant: "destructive",
        });
      }
    } catch (err) {
      console.error("Failed to delete conversation:", err);
      toast({
        title: "Error",
        description: "An error occurred while deleting the conversation",
        variant: "destructive",
      });
    }
  };

  // Handle clearing all conversations
  const handleClearAll = async () => {
    try {
      if (!Array.isArray(conversations) || conversations.length === 0) {
        setShowClearDialog(false);
        return;
      }
      
      let successCount = 0;
      let failCount = 0;
      
      // Since there's no API to delete all conversations at once, we'll delete them one by one
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
          title: "History Cleared",
          description: `Successfully deleted ${successCount} conversation(s)${failCount > 0 ? ` (Failed to delete ${failCount})` : ''}`,
        });
        
        // Clear conversations from local storage as well
        localStorage.removeItem('recent_conversations');
        
        // Update list
        refetch();
      } else if (failCount > 0) {
        toast({
          title: "Clear Failed",
          description: "No conversations were deleted, please try again",
          variant: "destructive",
        });
      }
    } catch (err) {
      console.error("Failed to clear conversations:", err);
      toast({
        title: "Error",
        description: "An error occurred while clearing conversations",
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
          <h3 className="text-sm font-medium">Recent Conversations</h3>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <Clock className="h-4 w-4" />
          </Button>
        </div>
        {[1, 2, 3].map((i) => (
          <div key={i} className="px-2">
            <div className="flex items-center space-x-2 p-2 rounded-md">
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
          Try Again
        </Button>
      </div>
    );
  }

  // No conversations state
  if (!Array.isArray(conversations) || conversations.length === 0) {
    return (
      <div className="px-4 py-2 text-sm text-center text-muted-foreground">
        No previous conversations
      </div>
    );
  }

  // Render conversation list
  return (
    <>
      <div className="space-y-1">
        <div className="flex justify-between items-center px-4 py-2">
          <h3 className="text-sm font-medium">Recent Conversations</h3>
          <div className="flex gap-1">
            {Array.isArray(conversations) && conversations.length > 0 && (
              <Button 
                variant="ghost" 
                size="sm" 
                className="text-destructive hover:text-destructive hover:bg-destructive/10"
                onClick={() => setShowClearDialog(true)}
                title="Clear History"
              >
                <TrashIcon className="h-4 w-4 mr-1" />
                <span className="text-xs">Clear All</span>
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={() => refetch()} title="Refresh">
              <Clock className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <ScrollArea className="max-h-[300px]">
          {Array.isArray(conversations) ? (
            conversations.map((conversation) => (
              <div key={conversation.id} className="flex justify-between items-center gap-2 px-3 py-2 hover:bg-muted/40 rounded-md mx-1 my-1">
                <div className="flex-1 overflow-hidden cursor-pointer" onClick={() => goToConversation(conversation.id)}>
                  <div className="flex items-start space-x-2">
                    <MessageSquare className="h-4 w-4 mt-1 flex-shrink-0 text-muted-foreground" />
                    <div className="flex flex-col items-start overflow-hidden">
                      <span className="text-sm font-medium truncate w-full text-start">
                        {conversation.title || 'Untitled conversation'}
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
                  title="Delete conversation"
                >
                  <Trash2 className="h-4 w-4 text-destructive" />
                </Button>
              </div>
            ))
          ) : (
            <div className="px-4 py-2 text-sm text-center text-muted-foreground">
              Unable to load conversations
            </div>
          )}
        </ScrollArea>
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
              onClick={handleClearAll}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Yes, Clear History
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
} 