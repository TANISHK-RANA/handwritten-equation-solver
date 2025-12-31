"""
Equation parser for converting recognized characters to mathematical results.

Handles parsing, validation, and safe evaluation of mathematical expressions.
"""

import re
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
    - Parentheses (if detected)
    """
    
    # Valid characters for equations
    VALID_CHARS = set('0123456789+-*/(). ')
    
    # Operator mapping for display
    OPERATOR_DISPLAY = {
        '*': '×',
        '/': '÷'
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
            Normalized equation string
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
        
        return normalized
    
    def validate_equation(self, equation: str) -> Tuple[bool, Optional[str]]:
        """
        Validate equation syntax.
        
        Args:
            equation: Normalized equation string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not equation:
            return False, "Empty equation"
        
        # Check for valid characters
        for char in equation:
            if char not in self.VALID_CHARS:
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
        
        # Check for consecutive operators
        if re.search(r'[+\-*/]{2,}', equation):
            return False, "Consecutive operators detected"
        
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
            
            # Use a restricted eval with only math operations
            # This is safe because we've already validated the input
            allowed_chars = set('0123456789+-*/.()')
            clean_equation = ''.join(c for c in equation if c in allowed_chars)
            
            if clean_equation != equation.replace(' ', ''):
                return None, "Invalid characters in equation"
            
            # Evaluate
            result = eval(clean_equation, {"__builtins__": {}}, {})
            
            # Round to reasonable precision
            if isinstance(result, float):
                result = round(result, 10)
            
            return float(result), None
            
        except ZeroDivisionError:
            return None, "Division by zero"
        except SyntaxError:
            return None, "Invalid equation syntax"
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
            equation: Normalized equation
            
        Returns:
            Display-formatted equation
        """
        display = equation
        for op, display_op in self.OPERATOR_DISPLAY.items():
            display = display.replace(op, f' {display_op} ')
        
        # Add spaces around + and -
        display = re.sub(r'\+', ' + ', display)
        display = re.sub(r'-', ' - ', display)
        
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
        "2+3",
        "10-4",
        "5*6",
        "20/4",
        "2+3*4",
        "10+5-3",
        "100/5+20",
        "2+3*4-8/2",
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

