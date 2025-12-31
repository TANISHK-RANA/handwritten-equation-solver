import React, { useRef, useState, useEffect, useCallback } from 'react';

/**
 * DrawingCanvas component for drawing handwritten equations
 * Supports both mouse and touch input
 */
export default function DrawingCanvas({ onImageCapture, disabled = false }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasContent, setHasContent] = useState(false);
  const [strokeHistory, setStrokeHistory] = useState([]);

  // Canvas settings
  const strokeWidth = 12;
  const strokeColor = '#ffffff';

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    
    if (!canvas || !container) return;

    // Set canvas size
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    
    // Set drawing styles
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = strokeWidth;
    ctx.strokeStyle = strokeColor;

    // Fill with black background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, rect.width, rect.height);
  }, []);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      
      if (!canvas || !container) return;

      // Save current image
      const imageData = canvas.toDataURL();
      
      // Resize canvas
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;

      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      
      // Restore styles
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.lineWidth = strokeWidth;
      ctx.strokeStyle = strokeColor;

      // Restore image
      const img = new Image();
      img.onload = () => {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, rect.width, rect.height);
        ctx.drawImage(img, 0, 0, rect.width, rect.height);
      };
      img.src = imageData;
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Get coordinates from event
  const getCoordinates = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    
    if (e.touches && e.touches.length > 0) {
      return {
        x: e.touches[0].clientX - rect.left,
        y: e.touches[0].clientY - rect.top
      };
    }
    
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }, []);

  // Start drawing
  const startDrawing = useCallback((e) => {
    if (disabled) return;
    
    e.preventDefault();
    const { x, y } = getCoordinates(e);
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    ctx.beginPath();
    ctx.moveTo(x, y);
    
    setIsDrawing(true);
  }, [disabled, getCoordinates]);

  // Draw
  const draw = useCallback((e) => {
    if (!isDrawing || disabled) return;
    
    e.preventDefault();
    const { x, y } = getCoordinates(e);
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    ctx.lineTo(x, y);
    ctx.stroke();
    
    setHasContent(true);
  }, [isDrawing, disabled, getCoordinates]);

  // Stop drawing
  const stopDrawing = useCallback((e) => {
    if (!isDrawing) return;
    
    e?.preventDefault();
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.closePath();
    
    // Save to history for undo
    setStrokeHistory(prev => [...prev, canvas.toDataURL()]);
    
    setIsDrawing(false);
  }, [isDrawing]);

  // Clear canvas
  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, rect.width, rect.height);
    
    setHasContent(false);
    setStrokeHistory([]);
  }, []);

  // Undo last stroke
  const undo = useCallback(() => {
    if (strokeHistory.length === 0) return;
    
    const newHistory = [...strokeHistory];
    newHistory.pop();
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    
    if (newHistory.length === 0) {
      // Clear to black
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, rect.width, rect.height);
      setHasContent(false);
    } else {
      // Restore previous state
      const img = new Image();
      img.onload = () => {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, rect.width, rect.height);
        ctx.drawImage(img, 0, 0, rect.width, rect.height);
      };
      img.src = newHistory[newHistory.length - 1];
    }
    
    setStrokeHistory(newHistory);
  }, [strokeHistory]);

  // Capture and send image
  const captureImage = useCallback(() => {
    if (!hasContent) return;
    
    const canvas = canvasRef.current;
    const imageData = canvas.toDataURL('image/png');
    
    if (onImageCapture) {
      onImageCapture(imageData);
    }
  }, [hasContent, onImageCapture]);

  return (
    <div className="flex flex-col gap-4 w-full">
      {/* Canvas Container */}
      <div 
        ref={containerRef}
        className="relative w-full aspect-[3/1] min-h-[150px] max-h-[250px] rounded-2xl overflow-hidden glow-border"
      >
        <canvas
          ref={canvasRef}
          className="drawing-canvas canvas-grid w-full h-full"
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
          onTouchCancel={stopDrawing}
        />
        
        {/* Placeholder text */}
        {!hasContent && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <p className="text-white/30 text-lg font-medium">
              Draw your equation here...
            </p>
          </div>
        )}
      </div>

      {/* Control Buttons */}
      <div className="flex flex-wrap gap-3 justify-center">
        <button
          onClick={undo}
          disabled={strokeHistory.length === 0 || disabled}
          className="btn-secondary flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
          </svg>
          Undo
        </button>
        
        <button
          onClick={clearCanvas}
          disabled={!hasContent || disabled}
          className="btn-danger flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
          Clear
        </button>
        
        <button
          onClick={captureImage}
          disabled={!hasContent || disabled}
          className="btn-primary flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
          Solve
        </button>
      </div>
    </div>
  );
}

