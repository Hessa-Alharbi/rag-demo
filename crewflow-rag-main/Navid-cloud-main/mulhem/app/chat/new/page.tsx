"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useFileUpload } from "@/hooks/useFileUpload";
import { useWebSocket } from "@/hooks/useWebSocket";
import apiClient from "@/lib/api-client";

export default function NewChat() {
  const router = useRouter();
  const {
    uploadFile,
    isLoading: uploadIsLoading,
    error: uploadError,
    uploadResponse,
    resetState,
  } = useFileUpload();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [chatTitle, setChatTitle] = useState("");
  const [taskId, setTaskId] = useState<string | null>(null);
  const [connectionFailed, setConnectionFailed] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadTimeout, setUploadTimeout] = useState<NodeJS.Timeout | null>(null);
  const {
    isConnected,
    message,
    error: wsError,
    sessionId,
    isInitialized,
  } = useWebSocket(taskId || "");

  // Watch for WebSocket status changes
  useEffect(() => {
    // Only redirect when we have a fully initialized task with a session ID
    if (isInitialized && message?.task_state === "SUCCESS" && sessionId) {
      if (uploadTimeout) {
        clearTimeout(uploadTimeout);
        setUploadTimeout(null);
      }
      router.push(`/chat/${sessionId}`);
    }
    
    // Set connection failed flag if WebSocket error occurs
    if (wsError && taskId) {
      console.error("WebSocket error:", wsError);
      setConnectionFailed(true);
    }
  }, [isInitialized, message, sessionId, router, wsError, taskId, uploadTimeout]);

  // Set up a timeout for uploads
  useEffect(() => {
    if (isUploading && !uploadTimeout) {
      const timeout = setTimeout(() => {
        if (isUploading) {
          setConnectionFailed(true);
          setIsUploading(false);
          console.error("Upload timed out after 300 seconds");
        }
      }, 300000); // 5 minute timeout
      setUploadTimeout(timeout);
    }

    return () => {
      if (uploadTimeout) {
        clearTimeout(uploadTimeout);
        setUploadTimeout(null);
      }
    };
  }, [isUploading, uploadTimeout]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      // Reset any previous errors when file is changed
      resetState();
      setConnectionFailed(false);
    }
  };

  const handleInitialize = async () => {
    if (!selectedFile) return;
    
    try {
      setIsUploading(true);
      console.log("Starting file upload...");
      const response = await uploadFile(selectedFile);
      console.log("Upload response:", response);
      
      // Check if we received a session_id directly
      if (response.session_id) {
        // We have a session ID, go directly to the chat
        router.push(`/chat/${response.session_id}`);
        setIsUploading(false);
        if (uploadTimeout) {
          clearTimeout(uploadTimeout);
          setUploadTimeout(null);
        }
      } else if (response.task_id) {
        // We have a task ID, set it for WebSocket monitoring
        setTaskId(response.task_id);
        
        // Using a more direct approach - navigate to the task ID page
        // which will handle the redirection once the task completes
        router.push(`/chat/${response.task_id}`);
      } else {
        // No valid ID in response
        throw new Error("Server did not return a valid session or task ID");
      }
    } catch (err: any) {
      console.error("Initialization failed:", err);
      setConnectionFailed(true);
      setIsUploading(false);
      if (uploadTimeout) {
        clearTimeout(uploadTimeout);
        setUploadTimeout(null);
      }
    }
  };

  const handleRetry = () => {
    // Reset states
    resetState();
    setConnectionFailed(false);
    setTaskId(null);
    setIsUploading(false);
    if (uploadTimeout) {
      clearTimeout(uploadTimeout);
      setUploadTimeout(null);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <Card>
        <CardHeader>
          <CardTitle>Initialize New Chat</CardTitle>
          <CardDescription>
            Upload a document to start a new conversation
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="title">Chat Title</Label>
            <Input
              id="title"
              placeholder="Enter chat title"
              value={chatTitle}
              onChange={(e) => setChatTitle(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="document">Upload Document</Label>
            <Input
              id="document"
              type="file"
              onChange={handleFileChange}
              accept=".doc,.docx,.pdf,.txt"
            />
          </div>
          
          {/* Error Alerts */}
          {(uploadError || (wsError && connectionFailed)) && (
            <Alert variant="destructive">
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>
                {uploadError?.message || wsError || "Chat initialization failed. Please try again."}
                <div className="mt-2">
                  <Button 
                    onClick={handleRetry}
                    className="flex items-center"
                  >
                    Try Again
                  </Button>
                </div>
              </AlertDescription>
            </Alert>
          )}
          
          {/* Status Message */}
          {isConnected && (
            <Alert>
              <AlertTitle>Status</AlertTitle>
              <AlertDescription>
                {message?.result?.state?.processing_status?.message || "Initializing chat session..."}
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
        <CardFooter>
          <Button
            className="w-full"
            onClick={handleInitialize}
            disabled={
              !selectedFile ||
              uploadIsLoading ||
              !chatTitle.trim() ||
              isConnected ||
              isUploading
            }
          >
            {uploadIsLoading || isUploading
              ? "Uploading..."
              : isConnected
              ? "Initializing Pipeline..."
              : "Initialize Chat"}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
