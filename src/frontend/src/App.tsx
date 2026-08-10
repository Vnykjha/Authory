import { useState, useEffect } from 'react';
import { analyzeEssay, healthCheck, type AnalyzeResponse, type SentenceResult } from './api/client';
import { EssayView } from './components/EssayView';
import { ExplanationPanel } from './components/ExplanationPanel';
import { LimitationsPanel } from './components/LimitationsPanel';
import { SampleEssays } from './components/SampleEssays';
import { Sparkles, Shield, AlertCircle, RefreshCw, ArrowRight } from 'lucide-react';

export function App() {
  const [essayText, setEssayText] = useState('');
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [selectedSentence, setSelectedSentence] = useState<SentenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);

  useEffect(() => {
    healthCheck().then(setApiConnected);
  }, []);

  const handleAnalyze = async (overrideText?: string) => {
    const textToAnalyze = overrideText || essayText;

    if (!textToAnalyze.trim() || textToAnalyze.length < 20) {
      setError('Please enter an essay of at least 20 characters.');
      return;
    }

    setLoading(true);
    setError(null);
    setSelectedSentence(null);

    try {
      const response = await analyzeEssay({ essay_text: textToAnalyze });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSampleSelect = (sampleText: string) => {
    setEssayText(sampleText);
    handleAnalyze(sampleText);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30">
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-indigo-600/15 rounded-full blur-3xl"></div>
        <div className="absolute top-1/3 -right-40 w-96 h-96 bg-rose-600/10 rounded-full blur-3xl"></div>
      </div>

      <nav className="relative z-10 border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-400">
              <Shield className="w-6 h-6" />
            </div>
            <div>
              <span className="font-extrabold text-xl tracking-tight text-slate-100 font-display">
                Authory<span className="text-indigo-400">.ai</span>
              </span>
              <span className="hidden sm:inline-block ml-3 px-2 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                Signal-Based Detection
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span className="text-slate-400 hidden sm:inline">Backend API:</span>
            {apiConnected === null ? (
              <span className="px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 border border-slate-700 animate-pulse">
                Connecting...
              </span>
            ) : apiConnected ? (
              <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-medium flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                Model Online
              </span>
            ) : (
              <span className="px-2.5 py-1 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/30 font-medium">
                Offline (Start API)
              </span>
            )}
          </div>
        </div>
      </nav>

      <main className="relative z-10 flex-1 max-w-5xl w-full mx-auto px-4 py-8 space-y-8">
        <section className="text-center space-y-3 max-w-2xl mx-auto">
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-100 font-display tracking-tight leading-tight">
            Signal-Based AI Detection for College Admissions
          </h1>
          <p className="text-sm sm:text-base text-slate-400 leading-relaxed">
            Decomposes essays into statistical perplexity, Binoculars cross-perplexity, burstiness, and lexical fingerprints — providing sentence-level highlighting and plain-language explanations.
          </p>
        </section>

        <section className="glass-panel p-6 rounded-2xl space-y-4 shadow-2xl">
          <div className="flex items-center justify-between">
            <label htmlFor="essay-input" className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              <span>Paste College Essay</span>
            </label>
            <span className="text-xs text-slate-400">
              {essayText.length} characters | {essayText.trim().split(/\s+/).filter(Boolean).length} words
            </span>
          </div>

          <textarea
            id="essay-input"
            value={essayText}
            onChange={(e) => setEssayText(e.target.value)}
            className="w-full h-44 p-4 bg-slate-900/80 border border-slate-700/60 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none resize-y text-slate-100 text-sm leading-relaxed font-sans transition-all placeholder:text-slate-500"
            placeholder="Paste your college essay text here to analyze..."
            disabled={loading}
          />

          <SampleEssays onSelectSample={handleSampleSelect} />

          <div className="pt-2 flex items-center justify-between gap-4">
            <button
              onClick={() => handleAnalyze()}
              disabled={loading || !essayText.trim()}
              className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 text-white font-semibold rounded-xl text-sm shadow-lg shadow-indigo-600/25 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Computing Signals...</span>
                </>
              ) : (
                <>
                  <span>Analyze Essay</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>

            {error && (
              <div className="flex items-center gap-1.5 text-xs text-rose-400 bg-rose-500/10 px-3 py-2 rounded-lg border border-rose-500/20">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </section>

        {result && (
          <section className="space-y-8 animate-fade-in">
            <EssayView
              sentences={result.sentences}
              summary={result.summary}
              processingTimeMs={result.processing_time_ms}
              selectedSentence={selectedSentence}
              onSentenceSelect={setSelectedSentence}
            />

            <LimitationsPanel eslFpr={0.436} nativeFpr={0.000} />

            <ExplanationPanel
              sentence={selectedSentence}
              onClose={() => setSelectedSentence(null)}
            />
          </section>
        )}
      </main>

      <footer className="relative z-10 border-t border-slate-800/80 py-6 text-center text-xs text-slate-500 space-y-1">
        <p>Authory AI Detector — Signal-Based Statistical Architecture</p>
        <p>Empirical Held-Out-Topic Evaluation &amp; ESL False-Positive Bias Transparency</p>
      </footer>
    </div>
  );
}

export default App;
