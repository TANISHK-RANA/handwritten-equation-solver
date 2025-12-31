import React from 'react';

/**
 * EquationHistory component for displaying previous equations
 */
export default function EquationHistory({ history, onSelect, onClear }) {
  if (history.length === 0) {
    return null;
  }

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white/80 flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          History
        </h3>
        <button
          onClick={onClear}
          className="text-sm text-accent-warning/70 hover:text-accent-warning transition-colors"
        >
          Clear all
        </button>
      </div>

      <div className="space-y-2 max-h-[300px] overflow-y-auto">
        {history.map((item, index) => (
          <button
            key={index}
            onClick={() => onSelect(item)}
            className="w-full text-left p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors group"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-white">
                {item.display_equation || item.recognized_equation}
              </span>
              {item.success && (
                <span className="font-mono text-accent-primary font-semibold">
                  = {item.formatted_result}
                </span>
              )}
            </div>
            <div className="text-xs text-white/40 mt-1">
              {new Date(item.timestamp).toLocaleTimeString()}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

