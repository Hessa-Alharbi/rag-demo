import { useState, useEffect } from "react";
import apiClient from "@/lib/api-client";

interface ProcessingStatus {
  status: "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED";
  message: string;
}

interface PipelineState {
  initialized: boolean;
  current_file: string;
  processing_status: ProcessingStatus;
  initialized_objects: any;
}

interface TaskResponse {
  status: string;
  result: {
    status: string;
    message: string;
    state: PipelineState;
    session_id: string;
  };
  session_id: string;
  pipeline_state: PipelineState;
}

export const usePipelineStatus = (taskId: string) => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<TaskResponse | null>(
    null
  );

  const checkStatus = async () => {
    try {
      const response = await apiClient.get<TaskResponse>(`/task/${taskId}`);
      setPipelineStatus(response.data);
      return response.data;
    } catch (err: any) {
      setError(err.message || "Failed to check pipeline status");
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (taskId) {
      const interval = setInterval(async () => {
        try {
          const status = await checkStatus();
          if (status.status === "SUCCESS" || status.status === "FAILED") {
            clearInterval(interval);
          }
        } catch (error) {
          clearInterval(interval);
        }
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [taskId]);

  return {
    isLoading,
    error,
    pipelineStatus,
    isPipelineReady:
      pipelineStatus?.status === "SUCCESS" &&
      pipelineStatus.pipeline_state.initialized &&
      pipelineStatus.pipeline_state.processing_status.status === "COMPLETED",
  };
};
