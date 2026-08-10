import React from 'react';
import type { SentenceResult } from '../api/client';
import { X, Sparkles, AlertCircle, Info, ChevronRight } from 'lucide-react';

interface ExplanationPanelProps {
  sentence: SentenceResult | null;
  onClose: () => void;
}

export const ExplanationPanel: React.FC<ExplanationPanelProps> = ({
  sentence,
  onClose,
}) => {
  if (!sentence) return null;

  const probPct = Math.round(sentence.ai_probability * 100);

  const getBadgeStyle = (pct: number) => {
    if (pct < 30) {
      return {
        badge: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
        bar: 'bg-emerald-500',
        label: 'Low AI Probability',
      };
    } else if (pct < 70) {
      return {
        badge: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
        bar: 'bg-amber-500',
        label: 'Moderate AI Signal',
      };
    } else {
      return {
        badge: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
        bar: 'bg-rose-500',
        label: 'High AI Probability',
      };
    }
  };

  const style = getBadgeStyle(probPct);

  return (
    <div className="fixed inset-y-0 right-0 z-50 w-full max-w-md bg-slate-900/95 backdrop-blur-xl border-l border-slate-800 shadow-2xl p-6 flex flex-col justify-between animate-slide-in">
      <div className="space-y-6 overflow-y-auto pr-1">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-indigo-400" />
            <h3 className="font-bold text-lg text-slate-100 font-display">
              Sentence #{sentence.sentence_idx + 1} Analysis
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            aria-label="Close explanation panel"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="bg-slate-800/60 p-4 rounded-xl border border-slate-700/50">
          <p className="text-xs uppercase tracking-wider text-slate-400 font-medium mb-1.5">Selected Passage</p>
          <p className="text-slate-200 italic leading-relaxed text-sm">
            "{sentence.text}"
          </p>
        </div>

        <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-800 space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-300 font-medium">{style.label}</span>
            <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold border ${style.badge}`}>
              {probPct}%
            </span>
          </div>
          <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${style.bar}`}
              style={{ width: `${probPct}%` }}
            />
          </div>
        </div>

        <div className="space-y-3">
          <h4 className="text-xs uppercase tracking-wider text-slate-400 font-semibold flex items-center gap-1.5">
            <AlertCircle className="w-4 h-4 text-indigo-400" />
            Top Contributing Signals
          </h4>

          {sentence.reasons.length === 0 || sentence.reasons[0] === 'standard natural language variation' ? (
            <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-xs text-emerald-300 flex items-start gap-2">
              <Info className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
              <span>No AI generation anomalies found — this sentence matches natural human syntax and perplexity variation.</span>
            </div>
          ) : (
            <div className="space-y-2.5">
              {sentence.reasons.map((reason, idx) => (
                <div
                  key={idx}
                  className="p-3.5 bg-slate-800/70 border border-slate-700/60 rounded-xl flex items-start gap-3 text-xs leading-relaxed text-slate-200"
                >
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-500/20 border border-indigo-500/40 text-indigo-300 text-xs font-bold flex items-center justify-center">
                    {idx + 1}
                  </span>
                  <div className="flex-1">
                    <p className="font-medium text-slate-200">{reason}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="p-3.5 bg-slate-900/80 rounded-xl border border-slate-800 text-[11px] text-slate-400 leading-relaxed">
          <p>
            Signals are computed from statistical language features (per-token logprobs, cross-perplexity ratios, burstiness metrics), not arbitrary LLM opinions.
          </p>
        </div>
      </div>

      <button
        onClick={onClose}
        className="mt-6 w-full py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-sm font-medium transition-colors flex items-center justify-center gap-1.5"
      >
        <span>Dismiss Panel</span>
        <ChevronRight className="w-4 h-4" />
      </button>
    </div>
  );
};
