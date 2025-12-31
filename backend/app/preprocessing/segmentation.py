"""
Character segmentation module for handwritten equation images.

Uses contour detection to segment individual characters from an equation image.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .image_utils import to_grayscale, apply_threshold, denoise_image


@dataclass
class BoundingBox:
    """Represents a character bounding box."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center_x(self) -> int:
        return self.x + self.width // 2
    
    @property
    def center_y(self) -> int:
        return self.y + self.height // 2
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class SegmentedCharacter:
    """Represents a segmented character with its image and position."""
    image: np.ndarray
    bounding_box: BoundingBox
    index: int  # Position in the equation (left to right)


class CharacterSegmenter:
    """
    Segments individual characters from a handwritten equation image.
    
    Uses contour detection and various heuristics to identify and extract
    individual characters while handling:
    - Multi-part characters (like = or :)
    - Overlapping characters
    - Variable spacing
    """
    
    def __init__(self, 
                 min_char_area: int = 20,
                 max_char_area: int = 50000,
                 min_aspect_ratio: float = 0.05,
                 max_aspect_ratio: float = 20.0,
                 merge_threshold: float = 0.3):
        """
        Initialize the segmenter.
        
        Args:
            min_char_area: Minimum area for a valid character
            max_char_area: Maximum area for a valid character
            min_aspect_ratio: Minimum width/height ratio
            max_aspect_ratio: Maximum width/height ratio
            merge_threshold: Threshold for merging nearby boxes (fraction of height)
        """
        self.min_char_area = min_char_area
        self.max_char_area = max_char_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.merge_threshold = merge_threshold
    
    def preprocess_for_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for contour detection.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Binary image ready for contour detection (white characters on black background)
        """
        # Convert to grayscale
        gray = to_grayscale(image)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Check if image is predominantly dark (canvas with white strokes)
        # or predominantly light (white paper with dark strokes)
        mean_val = np.mean(blurred)
        
        if mean_val < 128:
            # Dark background with light strokes (canvas format)
            # Just threshold without inversion
            _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        else:
            # Light background with dark strokes (scanned paper format)
            # Threshold and invert
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.bitwise_not(binary)
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Dilate slightly to connect broken strokes
        kernel_dilate = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel_dilate, iterations=1)
        
        return binary
    
    def find_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Find contours in the binary image.
        
        Args:
            binary_image: Binary image (white on black)
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(
            binary_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    
    def get_bounding_boxes(self, contours: List[np.ndarray]) -> List[BoundingBox]:
        """
        Get bounding boxes from contours and filter by size.
        
        Args:
            contours: List of contours
            
        Returns:
            List of valid bounding boxes
        """
        boxes = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box = BoundingBox(x, y, w, h)
            
            # Filter by area
            if box.area < self.min_char_area or box.area > self.max_char_area:
                continue
            
            # Filter by aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            boxes.append(box)
        
        return boxes
    
    def merge_overlapping_boxes(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Merge horizontally overlapping bounding boxes.
        
        This handles characters that might be split or multi-part symbols.
        
        Args:
            boxes: List of bounding boxes
            
        Returns:
            List of merged bounding boxes
        """
        if not boxes:
            return []
        
        # Sort by x position
        sorted_boxes = sorted(boxes, key=lambda b: b.x)
        merged = [sorted_boxes[0]]
        
        for box in sorted_boxes[1:]:
            last = merged[-1]
            
            # Check if boxes overlap horizontally
            overlap = (last.x + last.width) - box.x
            
            if overlap > 0 and overlap > self.merge_threshold * min(last.height, box.height):
                # Merge boxes
                new_x = min(last.x, box.x)
                new_y = min(last.y, box.y)
                new_right = max(last.x + last.width, box.x + box.width)
                new_bottom = max(last.y + last.height, box.y + box.height)
                
                merged[-1] = BoundingBox(
                    new_x, new_y,
                    new_right - new_x,
                    new_bottom - new_y
                )
            else:
                merged.append(box)
        
        return merged
    
    def sort_boxes_left_to_right(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Sort bounding boxes from left to right.
        
        Args:
            boxes: List of bounding boxes
            
        Returns:
            Sorted list of bounding boxes
        """
        return sorted(boxes, key=lambda b: b.center_x)
    
    def extract_character_images(self, 
                                 image: np.ndarray, 
                                 boxes: List[BoundingBox],
                                 padding: int = 5) -> List[np.ndarray]:
        """
        Extract character images from bounding boxes.
        
        Args:
            image: Original image
            boxes: List of bounding boxes
            padding: Padding to add around each character
            
        Returns:
            List of character images
        """
        h, w = image.shape[:2]
        characters = []
        
        for box in boxes:
            # Add padding
            x1 = max(0, box.x - padding)
            y1 = max(0, box.y - padding)
            x2 = min(w, box.x + box.width + padding)
            y2 = min(h, box.y + box.height + padding)
            
            # Extract character
            char_img = image[y1:y2, x1:x2]
            characters.append(char_img)
        
        return characters
    
    def segment(self, image: np.ndarray) -> List[SegmentedCharacter]:
        """
        Segment all characters from an equation image.
        
        Args:
            image: Input equation image
            
        Returns:
            List of SegmentedCharacter objects, sorted left to right
        """
        # Preprocess
        binary = self.preprocess_for_segmentation(image)
        
        # Find contours
        contours = self.find_contours(binary)
        
        # Get bounding boxes
        boxes = self.get_bounding_boxes(contours)
        
        # Merge overlapping boxes
        boxes = self.merge_overlapping_boxes(boxes)
        
        # Sort left to right
        boxes = self.sort_boxes_left_to_right(boxes)
        
        # Extract character images
        char_images = self.extract_character_images(binary, boxes)
        
        # Create SegmentedCharacter objects
        results = []
        for i, (img, box) in enumerate(zip(char_images, boxes)):
            results.append(SegmentedCharacter(
                image=img,
                bounding_box=box,
                index=i
            ))
        
        return results
    
    def visualize_segmentation(self, 
                               image: np.ndarray, 
                               segments: List[SegmentedCharacter]) -> np.ndarray:
        """
        Create a visualization of the segmentation results.
        
        Args:
            image: Original image
            segments: List of segmented characters
            
        Returns:
            Image with bounding boxes drawn
        """
        # Convert to color if grayscale
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        for seg in segments:
            box = seg.bounding_box
            # Draw rectangle
            cv2.rectangle(
                vis_image,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                (0, 255, 0),
                2
            )
            # Draw index
            cv2.putText(
                vis_image,
                str(seg.index),
                (box.x, box.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        return vis_image


def segment_equation(image: np.ndarray) -> List[np.ndarray]:
    """
    Convenience function to segment an equation image.
    
    Args:
        image: Input equation image
        
    Returns:
        List of character images, sorted left to right
    """
    segmenter = CharacterSegmenter()
    segments = segmenter.segment(image)
    return [seg.image for seg in segments]

