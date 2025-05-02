import apiClient from './api-client';

export interface ChatError {
  message: string;
  status: number;
}

export interface ChatRequest {
  query: string;
  session_id: string;
  history?: {
    type: 'human' | 'ai';
    content: string;
    timestamp: string;
  }[];
  max_new_tokens?: number;
  top_k?: number;
  top_p?: number;
  temperature?: number;
}

export interface ChatMessage {
  id: string;
  content: string;
  role: "user" | "assistant" | "system";
  created_at: string;
  context_docs?: Array<any>;
}

export interface ChatResponse {
  id: string;
  content?: string;
  created_at?: string;
  role?: string;
  // Add fields that might come from different API endpoints
  fused_answer?: string;
  answer?: string;
  response?: string;
  context_docs?: Array<{
    content: string;
    metadata?: {
      document_id?: string;
      title?: string;
      paragraph?: number;
      source?: string;
    };
    score?: number;
  }>;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  last_message?: string;
}

/**
 * Detect if the document is in Arabic
 */
function isArabicDocument(docTitle: string): boolean {
  return docTitle.endsWith('-ar.pdf') || 
         docTitle.includes('arabic') || 
         docTitle.includes('عربي') ||
         docTitle.includes('لائحة');
}

/**
 * Generate a fallback response
 */
function generateFallbackResponse(message: string, documentTitle: string): ChatResponse {
  // Is this an Arabic document?
  const isArabic = isArabicDocument(documentTitle);
  
  // Generate simple fallback response based on document type
  const content = isArabic 
    ? `لقد قمت بتحليل المستند "${documentTitle}". يرجى إخباري بالمعلومات المحددة التي تبحث عنها في هذا المستند.`
    : `I've analyzed the document "${documentTitle}". Please let me know what specific information you're looking for in this document.`;
  
  return {
    id: Math.random().toString(36).substring(2, 10),
    content,
    created_at: new Date().toISOString(),
    role: "assistant",
    context_docs: []
  };
}

/**
 * Send a chat message to the API
 */
export async function sendChatMessage(conversationId: string, message: string): Promise<ChatResponse> {
  try {
    console.log(`Sending message to conversation ${conversationId}`, message);
    
    // استدعاء نقطة نهاية API في الباكند - استخدام النظام الموجود فقط
    const response = await apiClient.post<any>('/query', {
      query: message,
      session_id: conversationId,
      history: [] 
    });
    
    console.log('Query endpoint response status:', response.status);
    
    if (response.data) {
      console.log('Response data from backend:', response.data);
      
      // استخراج الإجابة من الاستجابة
      const content = response.data.fused_answer || 
                     response.data.response || 
                     response.data.content || 
                     response.data.answer;
                     
      // استخراج context_docs من الاستجابة
      const contextDocs = response.data.docs || response.data.context_docs || [];
      
      if (content) {
        return {
          id: Math.random().toString(36).substring(2, 10),
          content: content,
          created_at: new Date().toISOString(),
          role: "assistant",
          // إضافة context_docs لاستخدامها في واجهة المستخدم
          context_docs: contextDocs
        };
      }
    }
    
    // إذا لم يتم العثور على محتوى، إرجاع رسالة افتراضية
    return {
      id: Math.random().toString(36).substring(2, 10),
      content: "لم أتمكن من العثور على إجابة مناسبة لسؤالك في المستند.",
      created_at: new Date().toISOString(),
      role: "assistant",
      context_docs: []
    };
    
  } catch (error: any) {
    console.error('API request failed:', error);
    
    // إرجاع رسالة خطأ
    return {
      id: Math.random().toString(36).substring(2, 10),
      content: "حدث خطأ أثناء معالجة طلبك. يرجى التحقق من اتصالك بالخادم والمحاولة مرة أخرى.",
      created_at: new Date().toISOString(),
      role: "assistant",
      context_docs: []
    };
  }
}

/**
 * Fetch recent conversations from the API
 */
export async function getRecentConversations(): Promise<Conversation[]> {
  try {
    console.log('Fetching recent conversations');
    
    // تصحيح مسار API - استخدام المسار الصحيح بدون مقطع chat/
    const response = await apiClient.get<Conversation[]>('/recent-conversations');
    
    console.log('Recent conversations response:', response.status);
    
    if (response.data) {
      console.log('Recent conversations data:', response.data);
      return response.data;
    }
    
    return [];
  } catch (error) {
    console.error('Error fetching recent conversations:', error);
    return [];
  }
}

/**
 * Delete a conversation
 */
export async function deleteConversation(conversationId: string): Promise<boolean> {
  try {
    console.log(`Deleting conversation ${conversationId}`);
    // استدعاء نقطة النهاية لحذف المحادثة
    const response = await apiClient.delete(`/chat/${conversationId}`);
    console.log('Delete conversation response:', response.status);
    return response.status === 200 || response.status === 204;
  } catch (error) {
    console.error('Error deleting conversation:', error);
    return false;
  }
}

/**
 * Clear all conversations
 */
export async function clearAllConversations(): Promise<{ success: boolean; count?: number }> {
  try {
    console.log('Clearing all conversations');
    // استدعاء نقطة النهاية لحذف جميع المحادثات
    const response = await apiClient.delete('/clear-conversations');
    console.log('Clear conversations response:', response.status);
    
    if (response.status === 200 || response.status === 204) {
      return { 
        success: true,
        count: response.data?.deleted_count || 0
      };
    }
    
    return { success: false };
  } catch (error) {
    console.error('Error clearing conversations:', error);
    return { success: false };
  }
}
