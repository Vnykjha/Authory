# Module 8: Frontend — Essay Rendering & Highlighting
**Day 9 — Tasks 9.2–9.5**  
**Estimated time:** 3–4 hours  
**Depends on:** Module 7 (`/analyze` endpoint working)  
**Can run in parallel with:** Module 9 (Explanation & Limitations Panels)

---

## Objective
Build a React + Tailwind frontend: paste box → "Analyze" → rendered essay with per-sentence color-coded highlights. Click/hover a sentence → shows details.

---

## Inputs Required
- Backend running at `http://localhost:8000` (or configurable API URL)
- `/analyze` endpoint returning `AnalyzeResponse` schema

---

## Outputs Produced
| Path | Description |
|------|-------------|
| `src/frontend/` | Complete React + Tailwind app |
| `src/frontend/package.json` | Dependencies |
| `src/frontend/src/App.tsx` | Main component |
| `src/frontend/src/components/EssayView.tsx` | Highlighted essay renderer |
| `src/frontend/src/components/SentenceSpan.tsx` | Individual sentence with highlight |
| `src/frontend/src/api/client.ts` | API client |

---

## Step-by-Step Tasks

### 9.2 Frontend Scaffold (Vite + React + TypeScript + Tailwind)

```bash
cd src/frontend
npm create vite@latest . -- --template react-ts
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**`tailwind.config.js`:**
```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // AI probability color scale: transparent (0) → warm red (1)
        aiProb: {
          0: 'rgba(255, 255, 255, 0)',
          10: 'rgba(255, 240, 240, 0.3)',
          20: 'rgba(255, 220, 220, 0.4)',
          30: 'rgba(255, 200, 200, 0.5)',
          40: 'rgba(255, 180, 180, 0.6)',
          50: 'rgba(255, 160, 160, 0.7)',
          60: 'rgba(255, 140, 140, 0.8)',
          70: 'rgba(255, 120, 120, 0.9)',
          80: 'rgba(255, 100, 100, 1)',
          90: 'rgba(230, 80, 80, 1)',
          100: 'rgba(200, 60, 60, 1)',
        }
      }
    },
  },
  plugins: [],
}
```

**`src/index.css`:**
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  html { @apply antialiased; }
  body { @apply bg-gray-50 text-gray-900; }
}

@layer components {
  .essay-container {
    @apply max-w-3xl mx-auto p-6 bg-white rounded-xl shadow-sm border border-gray-200;
  }
  .sentence-span {
    @apply inline-block px-1 py-0.5 rounded transition-colors duration-200 cursor-pointer;
  }
  .sentence-span:hover {
    @apply outline outline-2 outline-offset-1 outline-primary-500;
  }
}
```

---

### 9.3 API Client (`src/frontend/src/api/client.ts`)

```typescript
// src/frontend/src/api/client.ts
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
  avg_ai_probability: number;
  max_ai_probability: number;
  high_ai_sentences: string;
}

export interface AnalyzeResponse {
  essay_id: string;
  sentences: SentenceResult[];
  summary: EssaySummary;
  processing_time_ms: number;
}

export async function analyzeEssay(request: AnalyzeRequest): Promise<AnalyzeResponse> {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  
  return response.json();
}

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
```

---

### 9.4 Sentence Span Component (`src/frontend/src/components/SentenceSpan.tsx`)

```tsx
// src/frontend/src/components/SentenceSpan.tsx
import { SentenceResult } from '../api/client';

interface SentenceSpanProps {
  sentence: SentenceResult;
  onClick: (sentence: SentenceResult) => void;
  isSelected: boolean;
}

export function SentenceSpan({ sentence, onClick, isSelected }: SentenceSpanProps) {
  // Map probability 0-1 to Tailwind color class aiProb-{0-100}
  const probPct = Math.round(sentence.ai_probability * 100);
  const colorClass = `bg-aiProb-${probPct}`;
  
  return (
    <span
      className={`sentence-span ${colorClass} ${isSelected ? 'ring-2 ring-primary-500' : ''}`}
      onClick={() => onClick(sentence)}
      title={`AI probability: ${probPct}%`}
      data-sentence-idx={sentence.sentence_idx}
    >
      {sentence.text}
    </span>
  );
}
```

---

### 9.5 Essay View Component (`src/frontend/src/components/EssayView.tsx`)

```tsx
// src/frontend/src/components/EssayView.tsx
import { useState } from 'react';
import { SentenceResult, EssaySummary } from '../api/client';
import { SentenceSpan } from './SentenceSpan';

interface EssayViewProps {
  sentences: SentenceResult[];
  summary: EssaySummary;
  onSentenceSelect: (sentence: SentenceResult | null) => void;
  selectedSentence: SentenceResult | null;
}

export function EssayView({ sentences, summary, onSentenceSelect, selectedSentence }: EssayViewProps) {
  return (
    <div className="space-y-6">
      {/* Summary Banner */}
      <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
        <h3 className="font-semibold text-gray-900 mb-2">Overall Assessment</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Assessment:</span>
            <span className="ml-2 font-medium text-gray-900">{summary.qualitative_band}</span>
          </div>
          <div>
            <span className="text-gray-500">Avg AI Probability:</span>
            <span className="ml-2 font-medium text-gray-900">{(summary.avg_ai_probability * 100).toFixed(0)}%</span>
          </div>
          <div>
            <span className="text-gray-500">High-AI Sentences:</span>
            <span className="ml-2 font-medium text-gray-900">{summary.high_ai_sentences}</span>
          </div>
        </div>
      </div>
      
      {/* Essay Text with Highlights */}
      <div className="essay-container prose prose-gray max-w-none">
        <p className="whitespace-pre-wrap leading-relaxed text-lg">
          {sentences.map((sentence, idx) => (
            <SentenceSpan
              key={sentence.sentence_idx}
              sentence={sentence}
              onClick={onSentenceSelect}
              isSelected={selectedSentence?.sentence_idx === sentence.sentence_idx}
            />
          ))}
        </p>
      </div>
    </div>
  );
}
```

---

### 9.6 Main App (`src/frontend/src/App.tsx`)

```tsx
// src/frontend/src/App.tsx
import { useState } from 'react';
import { analyzeEssay, AnalyzeResponse, SentenceResult } from './api/client';
import { EssayView } from './components/EssayView';
import { ExplanationPanel } from './components/ExplanationPanel'; // Module 9
import { LimitationsPanel } from './components/LimitationsPanel'; // Module 9

function App() {
  const [essayText, setEssayText] = useState('');
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [selectedSentence, setSelectedSentence] = useState<SentenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const handleAnalyze = async () => {
    if (!essayText.trim() || essayText.length < 50) {
      setError('Please enter an essay (at least 50 characters)');
      return;
    }
    
    setLoading(true);
    setError(null);
    setSelectedSentence(null);
    
    try {
      const response = await analyzeEssay({ essay_text: essayText });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <header className="text-center">
          <h1 className="text-3xl font-bold text-gray-900">AI Essay Detector</h1>
          <p className="text-gray-600 mt-2">
            Signal-based analysis for college admissions essays — highlights AI-likely passages with explanations
          </p>
        </header>
        
        {/* Input */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold mb-4">Paste Your Essay</h2>
          <textarea
            value={essayText}
            onChange={(e) => setEssayText(e.target.value)}
            className="w-full h-48 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-y font-mono text-sm leading-relaxed"
            placeholder="Paste a college admissions essay here (50–10,000 characters)..."
            disabled={loading}
          />
          <div className="mt-4 flex items-center gap-4">
            <button
              onClick={handleAnalyze}
              disabled={loading || !essayText.trim()}
              className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
            {error && <span className="text-red-600 text-sm">{error}</span>}
          </div>
        </section>
        
        {/* Results */}
        {result && (
          <section className="space-y-6">
            <EssayView
              sentences={result.sentences}
              summary={result.summary}
              onSentenceSelect={setSelectedSentence}
              selectedSentence={selectedSentence}
            />
            
            {/* Explanation Panel (Module 9) */}
            <ExplanationPanel sentence={selectedSentence} onClose={() => setSelectedSentence(null)} />
            
            {/* Limitations Panel (Module 9) */}
            <LimitationsPanel />
          </section>
        )}
      </div>
    </div>
  );
}

export default App;
```

---

## Definition of Done (Module 8)
- [ ] `npm run dev` starts frontend at `http://localhost:5173` (or 3000)
- [ ] Paste box accepts text, "Analyze" calls `/analyze`
- [ ] Essay renders with per-sentence color highlights (intensity = AI probability)
- [ ] Click a sentence → selects it (visual ring), triggers explanation panel
- [ ] Summary banner shows qualitative band, avg/max probability, high-AI count
- [ ] Loading/error states handled
- [ ] No explanation panel or limitations panel yet (Module 9)

---

## Handoff to Next Modules
- **Module 9** needs: `selectedSentence` state, `EssayView` component structure
- Both modules can share the same `src/frontend/` directory