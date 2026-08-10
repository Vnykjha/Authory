import React from 'react';
import type { SentenceResult, EssaySummary } from '../api/client';
import { SentenceSpan } from './SentenceSpan';
import { ShieldCheck, AlertTriangle, ShieldAlert, Clock, Sparkles } from 'lucide-react';

interface EssayViewProps {
  sentences: SentenceResult[];
  summary: EssaySummary;
  processingTimeMs: number;
  selectedSentence: SentenceResult | null;
  onSentenceSelect: (sentence: SentenceResult | null) => void;
}

export const EssayView: React.FC<EssayViewProps> = ({
  sentences,
  summary,
  processingTimeMs,
  selectedSentence,
  onSentenceSelect,
}) => {
  const getBandBadge = (band: string) => {
    const lower = band.toLowerCase();
    if (lower.includes('human')) {
      return {
        bg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
        icon: <ShieldCheck className="w-5 h-5 text-emerald-400" />,
      };
    } else if (lower.includes('mixed') || lower.includes('places')) {
      return {
        bg: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
        icon: <AlertTriangle className="w-5 h-5 text-amber-400" />,
      };
    } else {
      return {
        bg: 'bg-rose-500/10 border-rose-500/30 text-rose-400',
        icon: <ShieldAlert className="w-5 h-5 text-rose-400" />,
      };
    }
  };

  const badgeInfo = getBandBadge(summary.qualitative_band);

  return (
    <div className="space-y-6">
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl border ${badgeInfo.bg}`}>
              {badgeInfo.icon}
            </div>
            <div>
              <p className="text-xs uppercase tracking-wider text-slate-400 font-medium">Overall Verdict</p>
              <h3 className="text-xl font-bold text-slate-100 font-display">
                {summary.qualitative_band}
              </h3>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs text-slate-400">
            <div className="flex items-center gap-1.5 bg-slate-800/60 px-3 py-1.5 rounded-lg border border-slate-700/50">
              <Clock className="w-4 h-4 text-indigo-400" />
              <span>{(processingTimeMs / 1000).toFixed(2)}s</span>
            </div>
            <div className="flex items-center gap-1.5 bg-slate-800/60 px-3 py-1.5 rounded-lg border border-slate-700/50">
              <Sparkles className="w-4 h-4 text-rose-400" />
              <span>{summary.high_ai_sentences_count}/{summary.total_sentences} High AI Sentences</span>
            </div>
          </div>
        </div>

        <p className="text-sm text-slate-300 leading-relaxed border-t border-slate-700/40 pt-4">
          {summary.summary_description}
        </p>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 bg-slate-900/60 rounded-xl border border-slate-800 text-xs">
        <span className="text-slate-400 font-medium">Click any sentence for detailed signal breakdown:</span>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-indigo-500/20 border border-indigo-500/40 inline-block"></span>
            <span className="text-slate-300">Natural Human (&lt; 30%)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-amber-500/40 border border-amber-500/60 inline-block"></span>
            <span className="text-slate-300">Moderate Signal (30–70%)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-rose-500/60 border border-rose-500 inline-block"></span>
            <span className="text-slate-300 font-medium">High AI Likelihood (&ge; 70%)</span>
          </div>
        </div>
      </div>

      <div className="glass-panel p-8 rounded-2xl leading-loose text-base text-slate-200 shadow-xl min-h-[250px]">
        <div className="whitespace-pre-wrap leading-relaxed space-y-4">
          {sentences.map((sent) => (
            <SentenceSpan
              key={sent.sentence_idx}
              sentence={sent}
              isSelected={selectedSentence?.sentence_idx === sent.sentence_idx}
              onSelect={onSentenceSelect}
            />
          ))}
        </div>
      </div>
    </div>
  );
};
