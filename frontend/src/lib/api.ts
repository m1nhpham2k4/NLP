import type {
  BuildIndexResponse,
  ChatMessage,
  ChatResponse,
  StatusResponse,
} from "../types";


const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1"
).replace(/\/$/, "");


async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      const fallback = await response.text();
      if (fallback) {
        message = fallback;
      }
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}


export const api = {
  getStatus(): Promise<StatusResponse> {
    return request<StatusResponse>("/status");
  },

  buildIndex(sources: string[]): Promise<BuildIndexResponse> {
    return request<BuildIndexResponse>("/index/build", {
      method: "POST",
      body: JSON.stringify({ sources }),
    });
  },

  sendChat(question: string, history: ChatMessage[]): Promise<ChatResponse> {
    return request<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify({
        question,
        history,
      }),
    });
  },
};
