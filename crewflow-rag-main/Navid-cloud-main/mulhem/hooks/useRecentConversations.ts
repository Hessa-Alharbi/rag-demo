import { useState, useEffect, useCallback } from 'react';
import { getRecentConversations, Conversation } from '@/lib/chat-service';

// Local storage key
const STORAGE_KEY = 'recent_conversations';

export function useRecentConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Retrieve conversations from local storage
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
      setConversations([]); // Ensure we always set a valid array
      return false;
    } catch (err) {
      console.error('Failed to load conversations from localStorage:', err);
      setConversations([]); // Ensure we always set a valid array
      return false;
    }
  }, []);

  // Save conversations to local storage
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
      
      // Attempt to fetch conversations from the backend
      const data = await getRecentConversations();
      
      if (Array.isArray(data) && data.length > 0) {
        setConversations(data);
        saveToLocalStorage(data);
      } else {
        // If unable to fetch conversations, load them from local storage
        if (!loadFromLocalStorage()) {
          setConversations([]);
        }
      }
    } catch (err) {
      console.error('Failed to fetch conversations:', err);
      setError('Failed to load recent conversations');
      
      // In case of error, try loading conversations from local storage
      if (!loadFromLocalStorage()) {
        setConversations([]); // Ensure we set a valid array if localStorage also fails
      }
    } finally {
      setIsLoading(false);
    }
  }, [loadFromLocalStorage, saveToLocalStorage]);

  useEffect(() => {
    // Start by retrieving conversations from local storage to improve user experience
    const hasLocalData = loadFromLocalStorage();
    
    // Then attempt to fetch updated data from the server
    fetchConversations();
    
    // Set up periodic updates (every 5 minutes)
    const intervalId = setInterval(fetchConversations, 5 * 60 * 1000);
    
    // Clean up timer when component is removed
    return () => clearInterval(intervalId);
  }, [fetchConversations, loadFromLocalStorage]);

  return {
    conversations,
    isLoading,
    error,
    refetch: fetchConversations
  };
} 