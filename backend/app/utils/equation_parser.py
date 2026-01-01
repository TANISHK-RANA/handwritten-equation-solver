"""
Equation parser for converting recognized characters to mathematical results.

Handles parsing, validation, and safe evaluation of mathematical expressions.
"""

import re
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ParseResult:
    """Result of equation parsing and evaluation."""
    equation: str  # Original recognized equation
    normalized: str  # Normalized equation string
    result: Optional[float]  # Computed result (None if invalid)
    error: Optional[str]  # Error message if any
    steps: List[str]  # Intermediate steps (for complex equations)
    
    @property
    def is_valid(self) -> bool:
        return self.error is None and self.result is not None


class EquationParser:
    """
    Safe equation parser and evaluator.
    
    Supports:
    - Basic arithmetic: +, -, *, /
    - Multi-digit numbers
    - Operator precedence (*, / before +, -)
    - Parentheses: (, )
    - Exponentiation: ^ (converted to **)
    - Square root: √ (converted to sqrt())
    """
    
    # Valid characters for equations (including new symbols)
    VALID_CHARS = set('0123456789+-*/().^ ')
    
    # Operator mapping for display
    OPERATOR_DISPLAY = {
        '*': '×',
        '/': '÷',
        '**': '^',
    }
    
    def __init__(self):
        """Initialize the parser."""
        pass
    
    def normalize_equation(self, equation: str) -> str:
        """
        Normalize equation string for evaluation.
        
        Args:
            equation: Raw equation string from recognition
            
        Returns:
            Normalized equation string for Python eval
        """
        # Remove whitespace
        normalized = equation.replace(' ', '')
        
        # Handle common recognition errors
        # 'x' -> '*' (multiplication)
        normalized = normalized.replace('x', '*')
        normalized = normalized.replace('X', '*')
        
        # '÷' -> '/' (division)
        normalized = normalized.replace('÷', '/')
        
        # '×' -> '*' (multiplication)
        normalized = normalized.replace('×', '*')
        
        # '^' -> '**' (exponentiation)
        normalized = normalized.replace('^', '**')
        
        # Handle square root: √number -> sqrt(number)
        # Pattern: √ followed by a number or parenthesized expression
        normalized = self._convert_sqrt(normalized)
        
        return normalized
    
    def _convert_sqrt(self, equation: str) -> str:
        """
        Convert √ notation to sqrt() function calls.
        
        Examples:
            √9 -> sqrt(9)
            √16 -> sqrt(16)
            √(4+5) -> sqrt(4+5)
        """
        result = equation
        
        # Handle √ followed by parentheses: √(expr) -> sqrt(expr)
        result = re.sub(r'√\(([^)]+)\)', r'sqrt(\1)', result)
        
        # Handle √ followed by a number: √9 -> sqrt(9)
        result = re.sub(r'√(\d+\.?\d*)', r'sqrt(\1)', result)
        
        return result
    
    def validate_equation(self, equation: str) -> Tuple[bool, Optional[str]]:
        """
        Validate equation syntax.
        
        Args:
            equation: Normalized equation string (after normalization, so uses ** not ^)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not equation:
            return False, "Empty equation"
        
        # Valid chars after normalization (includes ** for exponent, sqrt for square root)
        valid_chars_normalized = set('0123456789+-*/(). sqrtt')
        
        # Remove 'sqrt' temporarily to check other characters
        temp_eq = equation.replace('sqrt', '')
        
        # Check for valid characters
        for char in temp_eq:
            if char not in valid_chars_normalized:
                return False, f"Invalid character: {char}"
        
        # Check for balanced parentheses
        paren_count = 0
        for char in equation:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if paren_count < 0:
                return False, "Unbalanced parentheses"
        if paren_count != 0:
            return False, "Unbalanced parentheses"
        
        # Check for operators at start/end (except minus for negative numbers)
        if equation and equation[-1] in '+-*/':
            return False, "Equation ends with operator"
        if equation and equation[0] in '+*/':
            return False, "Equation starts with invalid operator"
        
        # Check for division by zero (simple check)
        if '/0' in equation and not re.search(r'/0[.]?\d', equation):
            return False, "Division by zero detected"
        
        return True, None
    
    def safe_eval(self, equation: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Safely evaluate a mathematical expression.
        
        Uses a restricted evaluation approach to prevent code injection.
        Supports: +, -, *, /, **, sqrt(), parentheses
        
        Args:
            equation: Normalized equation string
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            # Validate first
            is_valid, error = self.validate_equation(equation)
            if not is_valid:
                return None, error
            
            # Define safe math functions
            safe_dict = {
                "__builtins__": {},
                "sqrt": math.sqrt,
                "abs": abs,
                "pow": pow,
            }
            
            # Allowed characters in the expression (after normalization)
            # We allow: digits, operators, parentheses, dots, and 'sqrt'
            allowed_pattern = r'^[\d+\-*/().sqrt\s]+$'
            
            # Remove 'sqrt' for pattern check, add it back
            check_eq = equation.replace('sqrt', 'X')  # Replace with single char for check
            if not re.match(r'^[\d+\-*/().X\s]+$', check_eq):
                return None, "Invalid characters in equation"
            
            # Evaluate with safe context
            result = eval(equation, safe_dict, {})
            
            # Round to reasonable precision
            if isinstance(result, float):
                result = round(result, 10)
            
            return float(result), None
            
        except ZeroDivisionError:
            return None, "Division by zero"
        except SyntaxError:
            return None, "Invalid equation syntax"
        except ValueError as e:
            # Handle sqrt of negative numbers
            if "math domain error" in str(e):
                return None, "Cannot compute square root of negative number"
            return None, f"Value error: {str(e)}"
        except Exception as e:
            return None, f"Evaluation error: {str(e)}"
    
    def parse_and_evaluate(self, equation: str) -> ParseResult:
        """
        Parse and evaluate an equation string.
        
        Args:
            equation: Raw equation string from recognition
            
        Returns:
            ParseResult with equation, result, and any errors
        """
        # Normalize
        normalized = self.normalize_equation(equation)
        
        # Evaluate
        result, error = self.safe_eval(normalized)
        
        # Generate display steps for complex equations
        steps = self._generate_steps(normalized) if result is not None else []
        
        return ParseResult(
            equation=equation,
            normalized=normalized,
            result=result,
            error=error,
            steps=steps
        )
    
    def _generate_steps(self, equation: str) -> List[str]:
        """
        Generate intermediate calculation steps.
        
        Args:
            equation: Normalized equation
            
        Returns:
            List of steps
        """
        steps = [equation]
        
        # For simple equations, no intermediate steps needed
        if not any(op in equation for op in '+-*/'):
            return steps
        
        # Simple step generation: show multiplication/division first, then addition/subtraction
        # This is a simplified version - a full implementation would show actual intermediate results
        
        return steps
    
    def format_result(self, result: float) -> str:
        """
        Format result for display.
        
        Args:
            result: Numerical result
            
        Returns:
            Formatted string
        """
        if result == int(result):
            return str(int(result))
        else:
            # Format with up to 6 decimal places, removing trailing zeros
            formatted = f"{result:.6f}".rstrip('0').rstrip('.')
            return formatted
    
    def format_equation_for_display(self, equation: str) -> str:
        """
        Format equation for user-friendly display.
        
        Args:
            equation: Original equation (before normalization)
            
        Returns:
            Display-formatted equation with nice symbols
        """
        display = equation
        
        # Replace operators with display symbols
        display = display.replace('*', ' × ')
        display = display.replace('/', ' ÷ ')
        display = display.replace('^', '^')  # Keep ^ as is
        display = display.replace('√', '√')  # Keep √ as is
        
        # Add spaces around + and -
        display = re.sub(r'\+', ' + ', display)
        display = re.sub(r'(?<!\()(?<!\^)-', ' - ', display)  # Don't add space for negative numbers
        
        # Clean up multiple spaces
        display = re.sub(r'\s+', ' ', display).strip()
        
        return display


# Convenience function
def solve_equation(equation: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Convenience function to solve an equation.
    
    Args:
        equation: Equation string
        
    Returns:
        Tuple of (result, error)
    """
    parser = EquationParser()
    result = parser.parse_and_evaluate(equation)
    return result.result, result.error


if __name__ == "__main__":
    # Test the parser
    parser = EquationParser()
    
    test_equations = [
        # Basic arithmetic
        "2+3",
        "10-4",
        "5*6",
        "20/4",
        "2+3*4",
        "10+5-3",
        "100/5+20",
        "2+3*4-8/2",
        # Parentheses
        "(2+3)*4",
        "10/(2+3)",
        "(5+5)*(2+2)",
        # Exponentiation
        "2^3",
        "3^2+1",
        "2^(1+2)",
        "(2+1)^2",
        # Square root
        "√9",
        "√16+1",
        "√(4+5)",
        "2*√4",
        # Combined
        "2^3+√9",
        "(2+3)^2-√16",
    ]
    
    print("Testing Equation Parser:")
    print("-" * 50)
    
    for eq in test_equations:
        result = parser.parse_and_evaluate(eq)
        if result.is_valid:
            formatted = parser.format_result(result.result)
            print(f"{eq} = {formatted}")
        else:
            print(f"{eq} -> Error: {result.error}")

