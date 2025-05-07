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
 * Generate a safe random ID
 */
function generateSafeId(): string {
  try {
    return Math.random().toString(36).substring(2, 10);
  } catch (e) {
    // Fallback if toString or substring fails
    return Date.now().toString();
  }
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
    id: generateSafeId(),
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
    // Validate inputs
    if (!conversationId) {
      console.error('Missing conversation ID');
      throw new Error('Missing conversation ID');
    }

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
          id: generateSafeId(),
          content: content,
          created_at: new Date().toISOString(),
          role: "assistant",
          // إضافة context_docs لاستخدامها في واجهة المستخدم
          context_docs: contextDocs
        };
      }
    }
    
    // If no content was found, return a default message
    return {
      id: generateSafeId(),
      content: "I couldn't find a suitable answer to your question in the document.",
      created_at: new Date().toISOString(),
      role: "assistant",
      context_docs: []
    };
    
  } catch (error: any) {
    console.error('API request failed:', error);
    
    // Return error message
    return {
      id: generateSafeId(),
      content: "An error occurred while processing your request. Please check your connection to the server and try again.",
      created_at: new Date().toISOString(),
      role: "assistant",
      context_docs: []
    };
  }
}

/**
 * Initialize a new chat with a file
 */
export async function initializeChat(file: File): Promise<{ sessionId: string, conversationId: string }> {
  try {
    console.log(`Initializing chat with file ${file.name}`);
    
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await apiClient.post('/initialize', formData);
    
    if (response.data && response.data.session_id) {
      return {
        sessionId: response.data.session_id,
        conversationId: response.data.conversation_id || response.data.session_id
      };
    }
    
    throw new Error('Failed to initialize chat session');
  } catch (error) {
    console.error('Error initializing chat:', error);
    throw error;
  }
}

/**
 * Fetch recent conversations from the API
 */
export async function getRecentConversations(): Promise<Conversation[]> {
  try {
    console.log('Fetching recent conversations');
    
    // Correct API path for the recent conversations
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
    // Use the appropriate API endpoint for the backend
    const response = await apiClient.delete(`/conversations/${conversationId}`);
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
    // Call the correct endpoint to delete all conversations
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
