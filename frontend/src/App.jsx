import React, { useState, useEffect, useCallback } from 'react';
import DrawingCanvas from './components/DrawingCanvas';
import ResultDisplay from './components/ResultDisplay';
import EquationHistory from './components/EquationHistory';
import { solveEquation, checkHealth } from './api';

/**
 * Main App component for Handwritten Equation Solver
 */
export default function App() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [apiStatus, setApiStatus] = useState({ online: false, modelLoaded: false });

  // Check API status on mount
  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        const health = await checkHealth();
        setApiStatus({
          online: health.status === 'healthy',
          modelLoaded: health.model_loaded
        });
      } catch (err) {
        setApiStatus({ online: false, modelLoaded: false });
      }
    };

    checkApiStatus();
    // Check periodically
    const interval = setInterval(checkApiStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handle image capture from canvas
  const handleImageCapture = useCallback(async (imageData) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await solveEquation(imageData);
      
      // Add timestamp for history
      const resultWithTimestamp = {
        ...response,
        timestamp: Date.now()
      };
      
      setResult(response);
      
      // Add to history
      setHistory(prev => [resultWithTimestamp, ...prev].slice(0, 20));
      
    } catch (err) {
      console.error('Error solving equation:', err);
      setError(err.message || 'Failed to solve equation');
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Handle selecting from history
  const handleHistorySelect = useCallback((item) => {
    setResult(item);
    setError(null);
  }, []);

  // Clear history
  const handleClearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="py-6 px-4 border-b border-white/10">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-accent-primary to-accent-secondary rounded-xl">
              <svg className="w-8 h-8 text-canvas-bg" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl md:text-2xl font-bold text-white">
                Equation Solver
              </h1>
              <p className="text-xs text-white/50">Powered by CNN</p>
            </div>
          </div>

          {/* API Status */}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              apiStatus.online && apiStatus.modelLoaded ? 'bg-accent-primary animate-pulse' :
              apiStatus.online ? 'bg-accent-secondary' :
              'bg-accent-warning'
            }`}></div>
            <span className="text-xs text-white/50 hidden sm:block">
              {apiStatus.online && apiStatus.modelLoaded ? 'Ready' :
               apiStatus.online ? 'Model loading...' :
               'API offline'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 py-8 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Drawing Area - Takes 2 columns on large screens */}
            <div className="lg:col-span-2 space-y-8">
              {/* Canvas Section */}
              <section>
                <h2 className="text-lg font-semibold text-white/80 mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5 text-accent-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                  Draw Your Equation
                </h2>
                <DrawingCanvas 
                  onImageCapture={handleImageCapture}
                  disabled={isLoading || !apiStatus.online || !apiStatus.modelLoaded}
                />
                
                {/* Instructions */}
                <div className="mt-4 p-4 bg-white/5 rounded-xl">
                  <p className="text-sm text-white/60">
                    <span className="text-accent-primary font-medium">Tip:</span> Draw clear, large numbers and operators. 
                    Supported: digits (0-9) and operators (+, -, ×, ÷)
                  </p>
                </div>
              </section>

              {/* Result Section */}
              <section>
                <h2 className="text-lg font-semibold text-white/80 mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5 text-accent-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  Result
                </h2>
                <ResultDisplay 
                  result={result}
                  isLoading={isLoading}
                  error={error}
                />
              </section>
            </div>

            {/* Sidebar - History */}
            <div className="lg:col-span-1">
              <EquationHistory 
                history={history}
                onSelect={handleHistorySelect}
                onClear={handleClearHistory}
              />

              {/* API Status Card */}
              {!apiStatus.online && (
                <div className="glass-card p-6 mt-6 border-accent-warning/30">
                  <div className="flex items-start gap-3">
                    <div className="p-2 bg-accent-warning/20 rounded-lg">
                      <svg className="w-5 h-5 text-accent-warning" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                    </div>
                    <div>
                      <h4 className="text-accent-warning font-medium">API Offline</h4>
                      <p className="text-sm text-white/60 mt-1">
                        Please start the backend server:
                      </p>
                      <code className="block mt-2 text-xs bg-black/30 p-2 rounded text-accent-primary font-mono">
                        uvicorn app.main:app --reload
                      </code>
                    </div>
                  </div>
                </div>
              )}

              {apiStatus.online && !apiStatus.modelLoaded && (
                <div className="glass-card p-6 mt-6 border-accent-secondary/30">
                  <div className="flex items-start gap-3">
                    <div className="p-2 bg-accent-secondary/20 rounded-lg">
                      <svg className="w-5 h-5 text-accent-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div>
                      <h4 className="text-accent-secondary font-medium">Model Not Loaded</h4>
                      <p className="text-sm text-white/60 mt-1">
                        Train the model first:
                      </p>
                      <code className="block mt-2 text-xs bg-black/30 p-2 rounded text-accent-primary font-mono">
                        python training/train.py
                      </code>
                    </div>
                  </div>
                </div>
              )}

              {/* Info Card */}
              <div className="glass-card p-6 mt-6">
                <h4 className="font-medium text-white/80 mb-3">How it works</h4>
                <ol className="space-y-3 text-sm text-white/60">
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-5 h-5 bg-accent-primary/20 text-accent-primary rounded-full flex items-center justify-center text-xs font-bold">1</span>
                    <span>Draw your equation on the canvas</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-5 h-5 bg-accent-primary/20 text-accent-primary rounded-full flex items-center justify-center text-xs font-bold">2</span>
                    <span>CNN segments and recognizes each character</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-5 h-5 bg-accent-primary/20 text-accent-primary rounded-full flex items-center justify-center text-xs font-bold">3</span>
                    <span>Equation is parsed and evaluated</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-5 h-5 bg-accent-primary/20 text-accent-primary rounded-full flex items-center justify-center text-xs font-bold">4</span>
                    <span>Result displayed with confidence scores</span>
                  </li>
                </ol>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-4 px-4 border-t border-white/10">
        <div className="max-w-6xl mx-auto text-center text-sm text-white/40">
          <p>Handwritten Equation Solver • CNN-Powered • Built with TensorFlow & React</p>
        </div>
      </footer>
    </div>
  );
}

