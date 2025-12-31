import React from 'react';

/**
 * ResultDisplay component for showing equation results
 */
export default function ResultDisplay({ result, isLoading, error }) {
  if (isLoading) {
    return (
      <div className="glass-card p-8 flex flex-col items-center justify-center min-h-[200px]">
        <div className="spinner mb-4"></div>
        <p className="text-white/60 font-medium">Analyzing your equation...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-card p-8 border-accent-warning/30">
        <div className="flex items-start gap-4">
          <div className="p-3 bg-accent-warning/20 rounded-full">
            <svg className="w-6 h-6 text-accent-warning" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <div>
            <h3 className="text-accent-warning font-semibold text-lg mb-1">Error</h3>
            <p className="text-white/70">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="glass-card p-8 flex flex-col items-center justify-center min-h-[200px]">
        <div className="p-4 bg-white/5 rounded-full mb-4">
          <svg className="w-12 h-12 text-white/30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
        </div>
        <p className="text-white/40 text-center">
          Draw an equation above and click <span className="text-accent-primary">Solve</span> to see the result
        </p>
      </div>
    );
  }

  return (
    <div className="glass-card p-6 md:p-8 result-animate">
      {/* Equation Display */}
      <div className="mb-6">
        <span className="text-white/50 text-sm uppercase tracking-wider">Recognized Equation</span>
        <div className="mt-2 font-mono text-2xl md:text-3xl text-white">
          {result.display_equation || result.recognized_equation}
        </div>
      </div>

      {/* Result Display */}
      {result.success ? (
        <div className="bg-gradient-to-r from-accent-primary/10 to-accent-secondary/10 rounded-xl p-6 border border-accent-primary/20">
          <span className="text-accent-primary/70 text-sm uppercase tracking-wider">Result</span>
          <div className="mt-2 font-mono text-4xl md:text-5xl font-bold text-accent-primary glow-text typing-animation">
            = {result.formatted_result}
          </div>
        </div>
      ) : (
        <div className="bg-accent-warning/10 rounded-xl p-6 border border-accent-warning/20">
          <span className="text-accent-warning/70 text-sm uppercase tracking-wider">Could not evaluate</span>
          <div className="mt-2 text-white/70">
            {result.error || 'The equation could not be evaluated'}
          </div>
        </div>
      )}

      {/* Confidence Scores */}
      {result.confidence_scores && result.confidence_scores.length > 0 && (
        <div className="mt-6">
          <span className="text-white/50 text-sm uppercase tracking-wider">Character Confidence</span>
          <div className="mt-3 flex flex-wrap gap-2">
            {result.confidence_scores.map((item, index) => (
              <div 
                key={index}
                className="flex flex-col items-center p-2 bg-white/5 rounded-lg min-w-[50px]"
              >
                <span className="font-mono text-lg text-white">{item.char}</span>
                <span className={`text-xs ${
                  item.confidence > 0.9 ? 'text-accent-primary' :
                  item.confidence > 0.7 ? 'text-accent-secondary' :
                  'text-accent-warning'
                }`}>
                  {Math.round(item.confidence * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

