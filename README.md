# Handwritten Equation Solver

A machine learning application that recognizes and solves handwritten mathematical equations using Convolutional Neural Networks (CNN).

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![React](https://img.shields.io/badge/React-18+-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-99.49%25-brightgreen)

## Overview

This project provides a web-based interface where users can draw handwritten mathematical equations and get instant results. The system uses a Convolutional Neural Network (CNN) trained on the MNIST dataset (for digits 0-9) and synthetic operator symbols (+, -, *, /) to recognize 14 different characters with **99.49% accuracy**.

### Key Features

- **Drawing Canvas** - Draw equations directly in the browser with mouse or touch support
- **CNN Recognition** - Deep learning model trained on 70,000+ images
- **14 Class Classification** - Digits (0-9) and operators (+, -, *, /)
- **Real-time Results** - Instant equation solving with confidence scores
- **Responsive Design** - Works on desktop, tablet, and mobile devices
- **Pre-trained Model Included** - No training required to get started

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
   - [Clone Repository](#step-1-clone-the-repository)
   - [Backend Setup](#step-2-backend-setup)
   - [Frontend Setup](#step-3-frontend-setup)
3. [Running the Application](#running-the-application)
4. [Optional: Retrain Model](#optional-retrain-the-model)
5. [API Documentation](#api-documentation)
6. [Project Structure](#project-structure)
7. [Model Architecture](#model-architecture)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.8 or higher | `python --version` or `py -3 --version` |
| Node.js | 16 or higher | `node --version` |
| npm | 8 or higher | `npm --version` |
| Git | Any recent version | `git --version` |

### Platform-Specific Notes

- **Windows**: Use `py -3` instead of `python` if you have multiple Python versions installed
- **macOS/Linux**: Use `python3` if `python` points to Python 2.x

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/handwritten-equation-solver.git
cd handwritten-equation-solver
```

---

### Step 2: Backend Setup

#### 2.1 Navigate to Backend Directory

```bash
cd backend
```

#### 2.2 Create Virtual Environment

**Windows (Command Prompt):**
```cmd
py -3 -m venv venv
```

**Windows (PowerShell):**
```powershell
py -3 -m venv venv
```

**macOS/Linux:**
```bash
python3 -m venv venv
```

#### 2.3 Activate Virtual Environment

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

> **Note:** If you get a PowerShell execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**macOS/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt.

#### 2.4 Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (for CNN model)
- FastAPI (web framework)
- OpenCV (image processing)
- NumPy, SciPy, scikit-learn
- Uvicorn (ASGI server)

**Expected installation time:** 5-10 minutes (TensorFlow is large)

#### 2.5 Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

---

### Step 3: Frontend Setup

Open a **new terminal window** (keep the backend terminal for later).

#### 3.1 Navigate to Frontend Directory

```bash
cd handwritten-equation-solver/frontend
```

#### 3.2 Install Node.js Dependencies

```bash
npm install
```

**Expected installation time:** 1-2 minutes

---

## Running the Application

You need **two terminal windows** - one for backend, one for frontend.

### Terminal 1: Start Backend Server

```bash
cd handwritten-equation-solver/backend
```

Activate virtual environment (if not already active):
- **Windows:** `.\venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

Start the server:
```bash
uvicorn app.main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
Model loaded successfully from: ...\models\equation_solver_model.h5
```

**Backend URL:** http://localhost:8000

### Terminal 2: Start Frontend Server

```bash
cd handwritten-equation-solver/frontend
npm run dev
```

You should see:
```
VITE v4.x.x  ready in xxx ms
➜  Local:   http://localhost:3000/
```

**Frontend URL:** http://localhost:3000

### Using the Application

1. Open http://localhost:3000 in your browser
2. Draw a mathematical equation on the canvas (e.g., `5+2`, `12*3`)
3. Click **Solve** to get the result
4. View confidence scores for each recognized character

---

## Optional: Retrain the Model

The repository includes a **pre-trained model** (`backend/models/equation_solver_model.h5`) with 99.49% accuracy. Training is NOT required for normal use.

If you want to retrain the model:

```bash
cd backend

# Activate virtual environment first
# Windows: .\venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Train with default settings (30 epochs)
python training/train.py

# Or customize training
python training/train.py --epochs 50 --batch-size 64 --lr 0.0005
```

**Training details:**
- Downloads MNIST dataset automatically (~11MB)
- Generates synthetic operator symbols
- Training time: ~15-30 minutes on CPU, ~5 minutes on GPU
- Saves best model to `models/equation_solver_model.h5`

---

## API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with health status |
| `/health` | GET | API health check |
| `/solve` | POST | Solve equation from base64 image |
| `/upload` | POST | Upload image file and solve |
| `/model/status` | GET | Get model loading status |

### Example: Check API Health

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Example: Solve Equation

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,..."}'
```

Response:
```json
{
  "recognized_equation": "5+2",
  "result": 7.0,
  "formatted_result": "7",
  "confidence_scores": [
    {"char": "5", "confidence": 0.9987},
    {"char": "+", "confidence": 0.9994},
    {"char": "2", "confidence": 0.9991}
  ],
  "success": true
}
```

---

## Project Structure

```
handwritten-equation-solver/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application
│   │   ├── model/
│   │   │   ├── cnn_model.py        # CNN architecture definition
│   │   │   └── predictor.py        # Inference logic
│   │   ├── preprocessing/
│   │   │   ├── segmentation.py     # Character segmentation (OpenCV)
│   │   │   └── image_utils.py      # Image preprocessing utilities
│   │   └── utils/
│   │       └── equation_parser.py  # Equation parsing & evaluation
│   ├── training/
│   │   ├── train.py                # Model training script
│   │   ├── dataset.py              # Dataset loading (MNIST + operators)
│   │   └── augmentation.py         # Data augmentation
│   ├── models/
│   │   └── equation_solver_model.h5  # Pre-trained model (included)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DrawingCanvas.jsx   # Drawing canvas with touch support
│   │   │   ├── ResultDisplay.jsx   # Result display component
│   │   │   └── EquationHistory.jsx # History tracking
│   │   ├── App.jsx                 # Main application
│   │   ├── api.js                  # API communication
│   │   └── index.css               # Styles
│   ├── package.json
│   └── vite.config.js
├── research_paper/                  # IEEE research paper (if included)
├── .gitignore
└── README.md
```

---

## Model Architecture

The CNN model uses the following architecture:

```
Input Layer: 28x28x1 (grayscale image)
    │
    ▼
┌─────────────────────────────────────────────┐
│ Conv2D(32, 3x3) → BatchNorm → ReLU          │
│ Conv2D(32, 3x3) → BatchNorm → ReLU          │
│ MaxPooling2D(2x2) → Dropout(0.25)           │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Conv2D(64, 3x3) → BatchNorm → ReLU          │
│ Conv2D(64, 3x3) → BatchNorm → ReLU          │
│ MaxPooling2D(2x2) → Dropout(0.25)           │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Conv2D(128, 3x3) → BatchNorm → ReLU         │
│ MaxPooling2D(2x2) → Dropout(0.25)           │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Flatten                                      │
│ Dense(256) → BatchNorm → ReLU → Dropout(0.5)│
│ Dense(128) → BatchNorm → ReLU → Dropout(0.5)│
│ Dense(14) → Softmax                          │
└─────────────────────────────────────────────┘
    │
    ▼
Output: 14 classes [0-9, +, -, *, /]
```

### Training Results

| Class | Accuracy | Samples |
|-------|----------|---------|
| 0 | 99.59% | 980 |
| 1 | 99.38% | 1135 |
| 2 | 99.71% | 1032 |
| 3 | 99.80% | 1010 |
| 4 | 99.08% | 982 |
| 5 | 99.33% | 892 |
| 6 | 98.85% | 958 |
| 7 | 99.22% | 1028 |
| 8 | 99.69% | 974 |
| 9 | 99.31% | 1009 |
| + | 100.00% | 452 |
| - | 100.00% | 473 |
| * | 100.00% | 442 |
| / | 100.00% | 433 |
| **Overall** | **99.49%** | **11,800** |

---

## Troubleshooting

### Common Issues

#### 1. "Python not found" or wrong Python version

**Windows:**
```cmd
py -3 --version
```
Use `py -3` instead of `python` for all commands.

**macOS/Linux:**
```bash
python3 --version
```
Use `python3` instead of `python`.

#### 2. PowerShell Execution Policy Error

If you see "cannot be loaded because running scripts is disabled":
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 3. Port Already in Use

If port 8000 or 3000 is already in use:

**Backend (use different port):**
```bash
uvicorn app.main:app --reload --port 8001
```

**Frontend (edit vite.config.js):**
Change `port: 3000` to `port: 3001`

#### 4. Model Not Loading

If you see "Model not found":
- Ensure you're in the `backend` directory
- Check that `models/equation_solver_model.h5` exists
- If missing, retrain: `python training/train.py`

#### 5. CORS Error in Browser

The backend is configured to allow all origins. If you still get CORS errors:
- Ensure backend is running on port 8000
- Check browser console for the exact error

#### 6. TensorFlow Installation Issues

On some systems, TensorFlow may require additional setup:

**Windows:** Install Visual C++ Redistributable
**macOS (M1/M2):** Use `tensorflow-macos` instead

```bash
pip install tensorflow-macos  # For Apple Silicon
```

---

## Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| ML Framework | TensorFlow/Keras | 2.10+ |
| Backend | FastAPI | 0.100+ |
| Frontend | React + Vite | 18+ |
| Image Processing | OpenCV | 4.5+ |
| State Management | React Hooks | - |
| Styling | Tailwind CSS | 3.3+ |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - feel free to use and modify!

---

## Acknowledgments

- **MNIST Dataset** - Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **TensorFlow/Keras** - Google Brain Team
- **FastAPI** - Sebastián Ramírez
- **React** - Meta (Facebook)

---

## Contact

For questions or feedback, please open an issue on GitHub.
