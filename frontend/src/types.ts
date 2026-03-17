export type MessageRole = "user" | "assistant";

export interface ChatMessage {
  role: MessageRole;
  content: string;
}

export interface SearchResult {
  score: number;
  source: string;
  text: string;
  metadata: Record<string, string>;
}

export interface StatusResponse {
  app_name: string;
  index_ready: boolean;
  gemini_configured: boolean;
  query_rewrite_enabled: boolean;
  top_k: number;
  gemini_model: string;
  query_rewrite_model: string;
  embedding_model: string;
  index_path: string;
  embeddings_path: string;
  frontend_origins: string[];
}

export interface BuildIndexResponse {
  message: string;
  chunk_count: number;
  index_path: string;
  embeddings_path: string;
}

export interface ChatResponse {
  question: string;
  answer: string;
  citations: string;
  used_generation: boolean;
  rewritten_question: string;
  timestamp_utc: string;
  contexts: SearchResult[];
}

export interface LocalMessage extends ChatMessage {
  id: string;
}
