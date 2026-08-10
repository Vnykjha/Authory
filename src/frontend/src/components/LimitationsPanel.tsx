import React, { useState } from 'react';
import { AlertTriangle, ChevronDown, FileText, Layers } from 'lucide-react';

interface LimitationsPanelProps {
  eslFpr?: number;
  nativeFpr?: number;
}

export const LimitationsPanel: React.FC<LimitationsPanelProps> = ({
  eslFpr = 0.436,
  nativeFpr = 0.000,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="glass-panel rounded-2xl overflow-hidden transition-all duration-300">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full p-5 flex items-center justify-between text-left hover:bg-slate-800/40 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div>
            <h4 className="font-bold text-slate-100 font-display text-base">
              Known Limitations &amp; Model Architecture
            </h4>
            <p className="text-xs text-slate-400">
              Empirical bias disclosure (43.6% ESL False-Positive Rate) and signal methodology
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-amber-400 font-medium hidden sm:inline">
            {isOpen ? 'Collapse' : 'Expand Disclosure'}
          </span>
          <ChevronDown className={`w-5 h-5 text-slate-400 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} />
        </div>
      </button>

      {isOpen && (
        <div className="p-6 border-t border-slate-800 space-y-6 text-sm text-slate-300">
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-5 space-y-3">
            <div className="flex items-center gap-2 text-amber-400 font-bold text-sm">
              <AlertTriangle className="w-4 h-4" />
              <span>Empirical Bias Disclosure: Non-Native English (ESL) Essays</span>
            </div>

            <p className="text-xs leading-relaxed text-slate-300">
              Like all statistical perplexity detectors, Authory exhibits a measurable false-positive bias on non-native English writing.
            </p>

            <div className="grid grid-cols-2 gap-4 pt-1">
              <div className="bg-slate-900/80 p-3 rounded-lg border border-amber-500/20 text-center">
                <span className="text-2xl font-extrabold text-rose-400 font-display">{(eslFpr * 100).toFixed(1)}%</span>
                <p className="text-[11px] text-slate-400 mt-0.5">ESL False Positive Rate</p>
              </div>

              <div className="bg-slate-900/80 p-3 rounded-lg border border-emerald-500/20 text-center">
                <span className="text-2xl font-extrabold text-emerald-400 font-display">{(nativeFpr * 100).toFixed(1)}%</span>
                <p className="text-[11px] text-slate-400 mt-0.5">Native Human False Positive Rate</p>
              </div>
            </div>

            <p className="text-[11px] text-slate-400 leading-relaxed pt-1">
              <strong>Mechanism:</strong> Non-native writers naturally use simpler vocabulary and standard sentence structures. Statistical language models find standard word choices highly predictable (low perplexity), which detectors mistake for AI generation. Evaluators must never use this tool as sole proof of misconduct.
            </p>
          </div>

          <div className="space-y-3">
            <h5 className="font-semibold text-slate-100 text-sm flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-400" />
              How Authory Signal Detection Works
            </h5>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              <div className="bg-slate-800/40 p-3.5 rounded-xl border border-slate-700/40 space-y-1">
                <p className="font-medium text-slate-200">1. Single-Model Perplexity (GPT-2)</p>
                <p className="text-slate-400">Measures token log-probabilities and top predicted word frequencies (pct_rank_1).</p>
              </div>
              <div className="bg-slate-800/40 p-3.5 rounded-xl border border-slate-700/40 space-y-1">
                <p className="font-medium text-slate-200">2. Cross-Perplexity Ratio (Binoculars)</p>
                <p className="text-slate-400">Ratio between GPT-2 observer and GPT-2-medium performer predictions.</p>
              </div>
              <div className="bg-slate-800/40 p-3.5 rounded-xl border border-slate-700/40 space-y-1">
                <p className="font-medium text-slate-200">3. Burstiness &amp; Rhythm</p>
                <p className="text-slate-400">Variance (CV/IQR) of sentence lengths and perplexities across context windows.</p>
              </div>
              <div className="bg-slate-800/40 p-3.5 rounded-xl border border-slate-700/40 space-y-1">
                <p className="font-medium text-slate-200">4. Lexical Fingerprints</p>
                <p className="text-slate-400">Type-Token Ratio (TTR), MTLD richness, and stock AI transition phrase rates.</p>
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between border-t border-slate-800 pt-4 text-xs">
            <span className="text-slate-400">Read our full held-out-topic evaluation report:</span>
            <a
              href="file:///c:/Users/vnykj/OneDrive/Desktop/ai-essay-detector/docs/evaluation.md"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1.5 text-indigo-400 hover:text-indigo-300 font-medium transition-colors"
            >
              <FileText className="w-4 h-4" />
              <span>docs/evaluation.md</span>
            </a>
          </div>
        </div>
      )}
    </div>
  );
};
