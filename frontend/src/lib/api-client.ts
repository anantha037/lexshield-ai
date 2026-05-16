import axios, { AxiosError, AxiosInstance } from "axios";
import { getSession } from "next-auth/react";
import { ApiError } from "@/types";

const apiClient: AxiosInstance = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  timeout: 180000,
  headers: { "Content-Type": "application/json" },
});

apiClient.interceptors.request.use(async (config) => {
  const session = await getSession();
  if (session?.accessToken) {
    config.headers.Authorization = `Bearer ${session.accessToken}`;
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    const status = error.response?.status ?? 0;
    const apiError: ApiError = {
      status,
      message: "Something went wrong. Please try again.",
      isRateLimit: status === 429,
      isTimeout: status === 504 || error.code === "ECONNABORTED",
    };
    if (apiError.isRateLimit)
      apiError.message = "AI rate limit reached (Groq free tier). Please wait 30 seconds.";
    else if (apiError.isTimeout)
      apiError.message = "Request timed out. Server may be processing a large document.";
    else if (status === 401)
      apiError.message = "Session expired. Please log in again.";
    else if (status === 500)
      apiError.message = "Backend error. Check if FastAPI is running on :8000.";
    return Promise.reject(apiError);
  }
);

export const chatQuery = (payload: {
  query: string;
  session_id: string;
  run_rag?: boolean;
}) => apiClient.post("/api/v1/master/query", { run_rag: true, ...payload });

export const analyzeDocument = (formData: FormData) =>
  apiClient.post("/api/v1/master/document", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const getSessionHistory = (sessionId: string) =>
  apiClient.get(`/api/v1/master/session/${sessionId}/history`);

export const classifyDocument = (text: string) =>
  apiClient.post("/api/v1/classify", { text });

export default apiClient;