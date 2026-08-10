import React from 'react';
import type { SentenceResult } from '../api/client';

interface SentenceSpanProps {
  sentence: SentenceResult;
  isSelected: boolean;
  onSelect: (sentence: SentenceResult) => void;
}

function getHighlightStyle(probability: number): React.CSSProperties {
  if (probability < 0.30) {
    return {
      backgroundColor: 'rgba(99, 102, 241, 0.08)',
      color: '#e2e8f0',
      borderBottom: '1px dotted rgba(99, 102, 241, 0.3)',
    };
  } else if (probability < 0.70) {
    const alpha = 0.2 + (probability - 0.3) * 0.5;
    return {
      backgroundColor: `rgba(245, 158, 11, ${alpha})`,
      color: '#fff',
      borderBottom: '1px solid rgba(245, 158, 11, 0.6)',
    };
  } else {
    const alpha = 0.35 + (probability - 0.7) * 0.8;
    return {
      backgroundColor: `rgba(239, 68, 68, ${alpha})`,
      color: '#ffffff',
      fontWeight: 500,
      borderBottom: '2px solid rgba(239, 68, 68, 0.8)',
    };
  }
}

export const SentenceSpan: React.FC<SentenceSpanProps> = ({
  sentence,
  isSelected,
  onSelect,
}) => {
  const probPct = Math.round(sentence.ai_probability * 100);
  const style = getHighlightStyle(sentence.ai_probability);

  return (
    <span
      className={`sentence-span ${isSelected ? 'sentence-span-selected' : ''}`}
      style={style}
      onClick={() => onSelect(sentence)}
      title={`Sentence #${sentence.sentence_idx + 1} — AI Likelihood: ${probPct}% (Click for signals)`}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect(sentence);
        }
      }}
    >
      {sentence.text}{' '}
    </span>
  );
};
