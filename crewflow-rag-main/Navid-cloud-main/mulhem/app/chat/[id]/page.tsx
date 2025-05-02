"use client"

import { useState, useRef, useEffect } from "react"
import { useParams, useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Textarea } from "@/components/ui/textarea"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Card, CardContent } from "@/components/ui/card"
import { useWebSocket } from "@/hooks/useWebSocket"
import { sendChatMessage, ChatError } from '@/lib/chat-service'
import { MarkdownContent } from "@/components/markdown-content"
import { cn } from "@/lib/utils"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface Message {
  id: string
  role: "user" | "assistant" | "system"
  content: string
  timestamp?: string
  context_docs?: Array<any>
}

// مكون MessageDisplay مخصص لعرض الرسائل والمصادر
const MessageDisplay = ({ message }: { message: Message }) => {
  const isUser = message.role === "user";
  const hasContextDocs = message.context_docs && message.context_docs.length > 0;
  const contextDocsCount = message.context_docs?.length || 0;

  return (
    <Card
      className={cn(
        "p-4 mb-4",
        isUser ? "ml-auto bg-primary/10 max-w-[80%]" : "mr-auto bg-background max-w-[90%]"
      )}
    >
      <CardContent className="p-0 flex gap-3">
        {message.role !== "user" && (
          <Avatar className="h-8 w-8 flex-shrink-0">
            <AvatarFallback className="text-sm">AI</AvatarFallback>
            <AvatarImage src="/bot-avatar.png" alt="AI" />
          </Avatar>
        )}
        <div className={cn(
          "flex-1 overflow-hidden",
          isUser ? "text-right" : ""
        )}>
          <MarkdownContent content={message.content} />
          
          {/* عرض المصادر المرجعية إذا وجدت */}
          {hasContextDocs && (
            <div className="mt-4 pt-2 border-t">
              <details className="text-sm">
                <summary className="font-semibold cursor-pointer text-primary mb-2">
                  عرض المصادر ({contextDocsCount})
                </summary>
                <div className="overflow-auto max-h-80 text-xs bg-muted/20 p-2 rounded space-y-2">
                  {message.context_docs?.map((doc, index) => (
                    <div key={index} className="p-2 border border-border rounded-md">
                      <div className="flex justify-between items-start mb-1">
                        <div className="font-medium text-primary">المصدر {index + 1}</div>
                        {doc.metadata?.file_name && (
                          <div className="text-muted-foreground text-[10px]">{doc.metadata.file_name}</div>
                        )}
                      </div>
                      {doc.score && (
                        <div className="text-[10px] text-muted-foreground mb-1">
                          درجة التطابق: {Math.round(doc.score * 100)}%
                        </div>
                      )}
                      {doc.metadata?.title && (
                        <div className="text-[10px] text-muted-foreground mb-1">
                          القسم: {doc.metadata.title}
                        </div>
                      )}
                      <div className="whitespace-pre-wrap mt-1 bg-muted p-1 rounded text-[11px]">{doc.content}</div>
                    </div>
                  ))}
                </div>
              </details>
            </div>
          )}
        </div>
        {message.role === "user" && (
          <Avatar className="h-8 w-8 flex-shrink-0">
            <AvatarFallback className="text-sm">You</AvatarFallback>
            <AvatarImage src="/user-avatar.png" alt="User" />
          </Avatar>
        )}
      </CardContent>
    </Card>
  );
};

export default function ChatInterface() {
  const { id } = useParams()
  const router = useRouter()
  
  // Check if ID is a task ID (not a UUID format)
  const isTaskId = id && !String(id).includes('-')
  
  const {
    isConnected,
    message,
    error: wsError,
    isInitialized,
    sessionId
  } = useWebSocket(isTaskId ? (id as string) : '')

  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const [isInitializing, setIsInitializing] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  // مفتاح التخزين المحلي للمحادثة
  const getStorageKey = () => `chat_messages_${id}`;

  // حفظ المحادثة في التخزين المحلي
  const saveMessagesToLocalStorage = (msgs: Message[]) => {
    try {
      if (typeof window !== 'undefined' && id) {
        localStorage.setItem(getStorageKey(), JSON.stringify(msgs));
      }
    } catch (e) {
      console.error('Error saving messages to localStorage:', e);
    }
  };

  // استرجاع المحادثة من التخزين المحلي
  const loadMessagesFromLocalStorage = (): Message[] | null => {
    try {
      if (typeof window !== 'undefined' && id) {
        const saved = localStorage.getItem(getStorageKey());
        if (saved) {
          return JSON.parse(saved);
        }
      }
      return null;
    } catch (e) {
      console.error('Error loading messages from localStorage:', e);
      return null;
    }
  };

  // Initialize chat with welcome message
  useEffect(() => {
    console.log("Current ID:", id, "isTaskId:", isTaskId, "sessionId:", sessionId);
    
    // If we have a task ID and get back a session ID, redirect to the chat
    if (isTaskId && sessionId) {
      console.log("Redirecting from task to session:", sessionId);
      router.replace(`/chat/${sessionId}`);
      return;
    }

    // If we have a session ID directly, initialize the chat
    if (!isTaskId && id) {
      // محاولة تحميل المحادثة من التخزين المحلي أولاً
      const savedMessages = loadMessagesFromLocalStorage();
      
      if (savedMessages && savedMessages.length > 0) {
        console.log("Loaded chat messages from localStorage:", savedMessages.length);
        setMessages(savedMessages);
        setIsInitializing(false);
      } else {
        // إذا لم تكن هناك رسائل مخزنة، ابدأ محادثة جديدة
        setMessages([{ 
          id: "1", 
          role: "assistant", 
          content: `Chat session initialized. How can I help you today?` 
        }]);
        setIsInitializing(false);
      }
    }
    
    // Show loading message
    if (isInitializing && !error) {
      setMessages([{ 
        id: "loading", 
        role: "system", 
        content: `Initializing chat session. Please wait...` 
      }]);
    }

    // Set timeout for initialization
    const timeout = setTimeout(() => {
      if (isInitializing && !sessionId) {
        setError("Chat initialization timed out. Please try again.");
        setIsInitializing(false);
      }
    }, 30000); // 30 seconds timeout

    return () => clearTimeout(timeout);
  }, [id, isTaskId, sessionId, router, isInitializing, error]);

  // حفظ الرسائل عند تغييرها
  useEffect(() => {
    if (messages.length > 0 && !isInitializing && id) {
      saveMessagesToLocalStorage(messages);
    }
  }, [messages, isInitializing, id]);

  // Handle WebSocket errors
  useEffect(() => {
    if (wsError) {
      console.error("WebSocket error:", wsError);
      setError(`Connection error: ${wsError}`);
      setIsInitializing(false);
    }
  }, [wsError]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(scrollToBottom, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isTyping) return

    const userMessage: Message = {
      id: String(Date.now()),
      role: "user",
      content: input,
    }

    // Add user message to chat
    setMessages((prev) => [...prev, userMessage])
    
    // Clear input and show typing indicator
    const userInput = input.trim()
    setInput("")
    setIsTyping(true)
    
    // Reset error if any
    setError(null)

    try {
      console.log("Sending message to chat using RAG:", userInput)
      
      // تحويل المعرف إلى نوع نصي (string) وتحقق من قيمته
      const chatId = typeof id === 'string' ? id : Array.isArray(id) ? id[0] : String(id)
      
      if (!chatId) {
        throw new Error("Invalid chat session ID");
      }
      
      // إرسال الاستعلام إلى نظام RAG المتقدم
      const response = await sendChatMessage(chatId, userInput)
      
      console.log("Received response from RAG system:", response)
      
      // إضافة رد المساعد
      const assistantMessage: Message = {
        id: response.id || String(Date.now() + 1),
        role: "assistant",
        content: response.content || "I couldn't process your query. Please try again.",
        context_docs: response.context_docs || []
      }
      
      setMessages((prev) => [...prev, assistantMessage])
    } catch (err: any) {
      console.error("Chat error:", err)
      
      // عرض رسالة الخطأ في المحادثة
      setMessages((prev) => [
        ...prev,
        {
          id: String(Date.now() + 1),
          role: "system",
          content: `Error: ${err.message || "An error occurred while sending your message."}`,
        },
      ])
    } finally {
      setIsTyping(false)
    }
  }

  // If there's an error, we'll show it as a dismissable alert instead of replacing the whole UI
  const ErrorAlert = () => {
    if (!error) return null;
    
    return (
      <div className="fixed top-4 left-0 right-0 mx-auto w-full max-w-md z-50 p-4">
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
          <div className="mt-2 flex justify-end">
            <Button onClick={() => setError(null)}>Dismiss</Button>
          </div>
        </Alert>
      </div>
    );
  };

  // Show loading indicator while waiting for WebSocket connection
  if (isTaskId && !sessionId) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen p-4 text-center">
        <div className="w-16 h-16 border-4 border-t-primary rounded-full animate-spin mb-4"></div>
        <h2 className="text-xl font-semibold mb-2">Initializing chat session...</h2>
        <p className="text-muted-foreground">We're processing your document and setting up your chat session.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-screen max-h-screen">
      {/* Display error as a dismissable alert */}
      <ErrorAlert />
      
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full p-2">
          <div className="max-w-4xl mx-auto space-y-4 pb-20">
            {messages.map((message) => (
              <MessageDisplay key={message.id} message={message} />
            ))}
            {isTyping && (
              <Card className="mr-auto">
                <CardContent className="p-4 flex gap-3">
                  <Avatar>
                    <AvatarFallback>AI</AvatarFallback>
                    <AvatarImage src="/bot-avatar.png" />
                  </Avatar>
                  <div className="flex items-center">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>
      </div>
      
      <div className="border-t bg-background p-2">
        <form
          onSubmit={handleSubmit}
          className="max-w-4xl mx-auto flex items-end gap-2"
        >
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message here..."
            className="min-h-24 md:min-h-12 resize-none bg-background"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
          />
          <Button type="submit" disabled={isTyping}>
            Send
          </Button>
        </form>
      </div>
    </div>
  );
}

