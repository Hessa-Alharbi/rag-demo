import { useState, useEffect, useCallback } from 'react';
import { getRecentConversations, Conversation } from '@/lib/chat-service';

// مفتاح التخزين المحلي
const STORAGE_KEY = 'recent_conversations';

export function useRecentConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // استرجاع المحادثات من التخزين المحلي
  const loadFromLocalStorage = useCallback(() => {
    try {
      const storedData = localStorage.getItem(STORAGE_KEY);
      if (storedData) {
        const parsed = JSON.parse(storedData);
        if (Array.isArray(parsed)) {
          setConversations(parsed);
          return true;
        }
      }
      return false;
    } catch (err) {
      console.error('Failed to load conversations from localStorage:', err);
      return false;
    }
  }, []);

  // حفظ المحادثات في التخزين المحلي
  const saveToLocalStorage = useCallback((data: Conversation[]) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      return true;
    } catch (err) {
      console.error('Failed to save conversations to localStorage:', err);
      return false;
    }
  }, []);

  const fetchConversations = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // محاولة جلب المحادثات من الواجهة الخلفية
      const data = await getRecentConversations();
      
      if (data && data.length > 0) {
        setConversations(data);
        saveToLocalStorage(data);
      } else {
        // إذا لم تتمكن من جلب المحادثات، قم بتحميلها من التخزين المحلي
        if (!loadFromLocalStorage()) {
          setConversations([]);
        }
      }
    } catch (err) {
      console.error('Failed to fetch conversations:', err);
      setError('Failed to load recent conversations');
      
      // في حالة الخطأ، حاول تحميل المحادثات من التخزين المحلي
      loadFromLocalStorage();
    } finally {
      setIsLoading(false);
    }
  }, [loadFromLocalStorage, saveToLocalStorage]);

  useEffect(() => {
    // البدء باستعادة المحادثات من التخزين المحلي لتحسين تجربة المستخدم
    const hasLocalData = loadFromLocalStorage();
    
    // ثم محاولة جلب البيانات المحدثة من الخادم
    fetchConversations();
    
    // إعداد تحديث دوري (كل 5 دقائق)
    const intervalId = setInterval(fetchConversations, 5 * 60 * 1000);
    
    // تنظيف المؤقت عند إزالة المكون
    return () => clearInterval(intervalId);
  }, [fetchConversations, loadFromLocalStorage]);

  return {
    conversations,
    isLoading,
    error,
    refetch: fetchConversations
  };
} 