# Research Paper

This folder contains the IEEE format conference paper for the Handwritten Equation Solver project.

## Files

- `IEEE_Handwritten_Equation_Solver.tex` - LaTeX source file
- `figures/` - Folder for diagrams and figures

## Compiling the Paper

### Option 1: Overleaf (Recommended)
1. Go to [Overleaf](https://www.overleaf.com/)
2. Create a new project
3. Upload `IEEE_Handwritten_Equation_Solver.tex`
4. Select "IEEE" as the template if prompted
5. Click "Recompile" to generate PDF

### Option 2: Local LaTeX Installation

**Windows (MiKTeX):**
```bash
pdflatex IEEE_Handwritten_Equation_Solver.tex
bibtex IEEE_Handwritten_Equation_Solver
pdflatex IEEE_Handwritten_Equation_Solver.tex
pdflatex IEEE_Handwritten_Equation_Solver.tex
```

**macOS/Linux (TeX Live):**
```bash
pdflatex IEEE_Handwritten_Equation_Solver.tex
bibtex IEEE_Handwritten_Equation_Solver
pdflatex IEEE_Handwritten_Equation_Solver.tex
pdflatex IEEE_Handwritten_Equation_Solver.tex
```

## Before Submission

1. Replace `[Author Name]`, `[Department]`, `[University]`, etc. with your actual details
2. Add figures to the `figures/` folder
3. Update figure references in the LaTeX file
4. Review and update the abstract if needed
5. Check all references are correct

## Paper Structure

1. Abstract
2. Introduction
3. Literature Review
4. Methodology
5. Implementation
6. Results and Discussion
7. Comparison with Existing Solutions
8. Limitations
9. Future Work
10. Conclusion
11. References

