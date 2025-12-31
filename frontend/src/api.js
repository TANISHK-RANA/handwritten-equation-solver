/**
 * API service for communicating with the backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Solve an equation from a base64 encoded image
 * @param {string} imageBase64 - Base64 encoded image
 * @returns {Promise<Object>} Prediction result
 */
export async function solveEquation(imageBase64) {
  const response = await fetch(`${API_BASE_URL}/solve`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: imageBase64
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to solve equation');
  }

  return response.json();
}

/**
 * Check the health status of the API
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  
  if (!response.ok) {
    throw new Error('API is not responding');
  }

  return response.json();
}

/**
 * Get model status
 * @returns {Promise<Object>} Model status
 */
export async function getModelStatus() {
  const response = await fetch(`${API_BASE_URL}/model/status`);
  
  if (!response.ok) {
    throw new Error('Failed to get model status');
  }

  return response.json();
}

/**
 * Upload an image file and solve the equation
 * @param {File} file - Image file
 * @returns {Promise<Object>} Prediction result
 */
export async function uploadAndSolve(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to process uploaded file');
  }

  return response.json();
}

