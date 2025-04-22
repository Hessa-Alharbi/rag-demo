import { useEffect, useRef, useState } from 'react';

interface ProcessingStatus {
  status: string;
  message: string;
}

// Define WebSocket message structure
export interface WebSocketMessage {
  task_state: string;
  result?: {
    status: string;
    message: string;
    state: {
      initialized: boolean;
      current_file: string;
      processing_status: ProcessingStatus;
      initialized_objects: Record<string, any>;
    };
    session_id?: string;
  };
  session_id?: string;
  error?: string;
  pipeline_state?: any;
}

export const useWebSocket = (taskId: string) => {
  const ws = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [message, setMessage] = useState<WebSocketMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  useEffect(() => {
    // Only connect if we have a non-session ID
    if (!taskId || taskId.startsWith('chat_')) return;

    const connect = () => {
      // Update WebSocket URL to match the backend expectation
      const host = window.location.host;
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      
      // Direct connection to WebSocket without API prefix
      const wsUrl = `${protocol}//${host}/ws/task/${taskId}`;
      
      console.log(`Connecting to WebSocket: ${wsUrl}`);
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log('WebSocket connection established');
        setIsConnected(true);
        setError(null);
        reconnectAttempts.current = 0;
      };

      ws.current.onmessage = (event) => {
        try {
          console.log('WebSocket message received:', event.data);
          const data: WebSocketMessage = JSON.parse(event.data);
          setMessage(data);
        } catch (e) {
          console.error('Failed to parse WebSocket message', e);
          setError('Failed to parse WebSocket message');
        }
      };

      ws.current.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('WebSocket connection error');
        setIsConnected(false);
      };

      ws.current.onclose = (event) => {
        console.log('WebSocket connection closed', event);
        setIsConnected(false);
        
        // Try to reconnect if not closed normally and under max attempts
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current += 1;
          const timeout = Math.min(1000 * Math.pow(2, reconnectAttempts.current - 1), 10000);
          console.log(`Attempting to reconnect in ${timeout}ms (attempt ${reconnectAttempts.current})`);
          setTimeout(connect, timeout);
        } else if (reconnectAttempts.current >= maxReconnectAttempts) {
          setError('Failed to connect after multiple attempts');
        }
      };
    };

    connect();

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [taskId]);

  return {
    isConnected,
    message,
    error,
    sessionId: message?.session_id,
    isInitialized: 
      message?.task_state === 'SUCCESS' && 
      message?.result?.state?.initialized && 
      message?.result?.state?.processing_status?.status === 'COMPLETED'
  };
};
