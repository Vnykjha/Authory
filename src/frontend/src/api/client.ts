/**
 * API Client for Authory AI Essay Detector backend.
 */

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export interface AnalyzeRequest {
  essay_text: string;
  essay_id?: string;
  topic?: string;
}

export interface SentenceResult {
  sentence_idx: number;
  text: string;
  start_char: number;
  end_char: number;
  ai_probability: number;
  reasons: string[];
}

export interface EssaySummary {
  qualitative_band: string;
  summary_description: string;
  avg_ai_probability: number;
  max_ai_probability: number;
  high_ai_sentences_count: number;
  total_sentences: number;
}

export interface AnalyzeResponse {
  essay_id: string;
  sentences: SentenceResult[];
  summary: EssaySummary;
  processing_time_ms: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) return false;
    const data: HealthResponse = await response.json();
    return data.status === 'ok' && data.model_loaded;
  } catch {
    return false;
  }
}

export async function analyzeEssay(request: AnalyzeRequest): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Analysis server error' }));
    throw new Error(errorData.detail || `HTTP ${response.status}: Failed to analyze essay`);
  }

  return response.json();
}
