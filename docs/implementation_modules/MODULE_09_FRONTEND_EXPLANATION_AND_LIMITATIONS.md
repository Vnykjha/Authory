# Module 9: Frontend — Explanation & Limitations Panels
**Day 9 — Task 9.6**  
**Estimated time:** 2–3 hours  
**Depends on:** Module 8 (EssayView, selectedSentence state)  
**Can run in parallel with:** Module 8

---

## Objective
Build two side panels: (1) Explanation Panel — shows top reasons for a selected sentence's AI probability; (2) Limitations Panel — displays "How this works" + measured ESL false-positive gap in plain language.

---

## Inputs Required
- Module 8's `selectedSentence` state (contains `reasons` array, `ai_probability`)
- Backend `/analyze` already returns reasons
- ESL FPR numbers from `docs/evaluation.md` (or hardcoded for now, wired to Module 10 later)

---

## Outputs Produced
| Path | Description |
|------|-------------|
| `src/frontend/src/components/ExplanationPanel.tsx` | Click/hover panel with reasons |
| `src/frontend/src/components/LimitationsPanel.tsx` | "How this works" + "Known limitations" with ESL gap |

---

## Step-by-Step Tasks

### 9.6.1 Explanation Panel (`src/frontend/src/components/ExplanationPanel.tsx`)

```tsx
// src/frontend/src/components/ExplanationPanel.tsx
import { SentenceResult } from '../api/client';

interface ExplanationPanelProps {
  sentence: SentenceResult | null;
  onClose: () => void;
}

export function ExplanationPanel({ sentence, onClose }: ExplanationPanelProps) {
  if (!sentence) return null;
  
  const probPct = Math.round(sentence.ai_probability * 100);
  const getProbColor = (p: number) => {
    if (p < 30) return 'text-green-700 bg-green-50 border-green-200';
    if (p < 70) return 'text-yellow-700 bg-yellow-50 border-yellow-200';
    return 'text-red-700 bg-red-50 border-red-200';
  };
  
  return (
    <div className="fixed right-4 top-20 bottom-4 w-80 bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden flex flex-col z-50 animate-slide-in">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <h3 className="font-semibold text-gray-900">Sentence Analysis</h3>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-xl leading-none">×</button>
      </div>
      
      {/* Sentence Text */}
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <p className="text-sm text-gray-600 mb-2">Sentence #{sentence.sentence_idx + 1}</p>
        <p className="text-gray-900 whitespace-pre-wrap">{sentence.text}</p>
      </div>
      
      {/* Probability Badge */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">AI Likelihood</span>
          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getProbColor(probPct)}`}>
            {probPct}%
          </span>
        </div>
        <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
          <div 
            className={`h-full rounded-full transition-all duration-500 ${probPct < 30 ? 'bg-green-500' : probPct < 70 ? 'bg-yellow-500' : 'bg-red-500'}`}
            style={{ width: `${probPct}%` }}
          />
        </div>
      </div>
      
      {/* Reasons */}
      <div className="flex-1 p-4 overflow-y-auto">
        <h4 className="font-medium text-gray-900 mb-3">Contributing Signals</h4>
        {sentence.reasons.length === 0 || sentence.reasons[0] === "no strong signals" ? (
          <p className="text-gray-500 text-sm italic">
            No strong signals detected — this sentence appears consistent with human writing patterns.
          </p>
        ) : (
          <ul className="space-y-2">
            {sentence.reasons.map((reason, idx) => (
              <li key={idx} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary-100 text-primary-700 text-xs font-bold flex items-center justify-center">
                  {idx + 1}
                </span>
                <span className="text-sm text-gray-700 leading-relaxed">{reason}</span>
              </li>
            ))}
          </ul>
        )}
        
        <div className="mt-4 pt-4 border-t border-gray-200 text-xs text-gray-500">
          <p>These signals are derived from statistical patterns in language model predictions, not from an AI "opinion." See <a href="#" className="text-primary-600 hover:underline">How this works</a> for details.</p>
        </div>
      </div>
    </div>
  );
}
```

**Add animation to `src/index.css`:**
```css
@keyframes slide-in {
  from { opacity: 0; transform: translateX(20px); }
  to { opacity: 1; transform: translateX(0); }
}
.animate-slide-in { animation: slide-in 0.2s ease-out; }
```

---

### 9.6.2 Limitations Panel (`src/frontend/src/components/LimitationsPanel.tsx`)

```tsx
// src/frontend/src/components/LimitationsPanel.tsx
import { useState } from 'react';

interface LimitationsPanelProps {
  eslFpr?: number;        // e.g., 0.42 for 42%
  nativeFpr?: number;     // e.g., 0.08 for 8%
}

export function LimitationsPanel({ eslFpr = 0.42, nativeFpr = 0.08 }: LimitationsPanelProps) {
  const [expanded, setExpanded] = useState(false);
  
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <span className="font-semibold text-gray-900 flex items-center gap-2">
          <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Known Limitations & How This Works
        </span>
        <svg 
          className={`w-5 h-5 text-gray-500 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      
      {expanded && (
        <div className="px-6 pb-6 border-t border-gray-200 space-y-6">
          {/* How It Works */}
          <section>
            <h4 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
              <svg className="w-5 h-5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              How This Works
            </h4>
            <div className="space-y-2 text-sm text-gray-700">
              <p>This detector does <strong>not</strong> ask an AI model "was this written by AI?" — that approach is unreliable and cannot explain its reasoning.</p>
              <p>Instead, it runs your essay through <strong>two small language models</strong> (GPT-2 family) and measures statistical signals:</p>
              <ul className="list-disc list-inside space-y-1 ml-4">
                <li><strong>Perplexity:</strong> How "surprised" a model is by each word choice. AI text tends to use highly predictable words.</li>
                <li><strong>Cross-Perplexity (Binoculars):</strong> Ratio between two models' predictions — more robust across writing styles.</li>
                <li><strong>Burstiness:</strong> Variation in sentence length and unpredictability. Human writing varies; AI tends to be consistent.</li>
                <li><strong>Lexical Fingerprints:</strong> Vocabulary diversity and frequency of stock AI transition phrases.</li>
              </ul>
              <p>A logistic regression classifier (trained on labeled human/AI/hybrid essays) combines these signals into a per-sentence AI probability with plain-language explanations.</p>
            </div>
          </section>
          
          {/* Known Limitations */}
          <section>
            <h4 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
              <svg className="w-5 h-5 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              Known Limitations
            </h4>
            <div className="space-y-3 text-sm text-gray-700">
              <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
                <p className="font-medium text-amber-800 mb-2">⚠️ Bias Against Non-Native English Writers</p>
                <p>Like most AI detectors, this tool is <strong>measurably less reliable on non-native English writing</strong>.</p>
                <div className="mt-3 grid grid-cols-2 gap-4 text-center">
                  <div className="bg-white p-3 rounded border border-amber-200">
                    <p className="text-2xl font-bold text-amber-700">{(eslFpr * 100).toFixed(0)}%</p>
                    <p className="text-xs text-amber-600">False Positive Rate (ESL)</p>
                  </div>
                  <div className="bg-white p-3 rounded border border-amber-200">
                    <p className="text-2xl font-bold text-amber-700">{(nativeFpr * 100).toFixed(0)}%</p>
                    <p className="text-xs text-amber-600">False Positive Rate (Native)</p>
                  </div>
                </div>
                <p className="mt-3 text-xs text-amber-700">
                  <strong>Why:</strong> Non-native writers often use simpler, more standard vocabulary and sentence structures — which produces lower perplexity, the same statistical signature the detector uses to flag AI text. This is a known limitation of <em>all</em> perplexity-based detectors (Liang et al., 2023).
                </p>
              </div>
              
              <ul className="list-disc list-inside space-y-1 ml-4">
                <li><strong>Hybrid essays are hardest:</strong> Human drafts with AI polishing passes may show mixed signals.</li>
                <li><strong>Short texts are unreliable:</strong> Essays under ~200 words have high variance.</li>
                <li><strong>Topic generalization:</strong> Accuracy drops on topics not seen during training (held-out-topic test: see evaluation doc).</li>
                <li><strong>Adversarial edits:</strong> Deliberate human edits to "sound more AI-like" (or vice versa) can fool the detector.</li>
                <li><strong>Model coverage:</strong> Trained on specific AI models; may not generalize to newer/larger models.</li>
              </ul>
            </div>
          </section>
          
          {/* Evaluation Link */}
          <section>
            <h4 className="font-medium text-gray-900 mb-2">Full Evaluation Report</h4>
            <p className="text-sm text-gray-700">
              See <a href="/evaluation.md" target="_blank" rel="noopener" className="text-primary-600 hover:underline font-medium">
                docs/evaluation.md
              </a> for confusion matrices, held-out-topic results, three confidently-wrong examples with hypotheses, and the complete ESL bias analysis.
            </p>
          </section>
        </div>
      )}
    </div>
  );
}
```

---

### 9.6.3 Wire Into App (update `App.tsx` from Module 8)

```tsx
// In App.tsx, add imports:
import { ExplanationPanel } from './components/ExplanationPanel';
import { LimitationsPanel } from './components/LimitationsPanel';

// In the results section, replace the placeholder comments:
{result && (
  <section className="space-y-6">
    <EssayView ... />
    
    {/* Explanation Panel - appears on right when sentence selected */}
    <ExplanationPanel 
      sentence={selectedSentence} 
      onClose={() => setSelectedSentence(null)} 
    />
    
    {/* Limitations Panel - always visible at bottom */}
    <LimitationsPanel 
      eslFpr={0.42}   // Replace with real numbers from Module 10
      nativeFpr={0.08}
    />
  </section>
)}
```

---

## Definition of Done (Module 9)
- [ ] `ExplanationPanel` appears on sentence click, shows probability badge + top 3 reasons
- [ ] Panel animates in/out, closes on × or clicking another sentence
- [ ] `LimitationsPanel` collapses/expands, shows "How this works" + ESL bias warning
- [ ] ESL FPR numbers displayed prominently (placeholder values for now)
- [ ] Link to full evaluation document
- [ ] No bare percentages in summary — qualitative bands only (already done in Module 8)
- [ ] Responsive: panels stack on mobile, side-by-side on desktop

---

## Handoff to Next Modules
- **Module 10** needs: real ESL/native FPR numbers to replace placeholders in `LimitationsPanel`
- Frontend complete — Module 10 is documentation/polish only