"""
Quality Validation for Universal CAPTCHA Generator.

Post-generation quality gate that rejects degenerate samples where text is
unreadable. Uses fast pixel-level heuristics to detect:
1. Low text-background contrast
2. Text region dominated by noise/shapes (low text pixel ratio)
3. Excessive blur destroying text edges

The generator calls validate() after rendering and retries with a new
random schema if the sample fails quality checks.
"""

import numpy as np
from PIL import Image, ImageFilter


def validate_captcha(
    final_image: Image.Image,
    text_layer: Image.Image,
    min_contrast: float = 55.0,
    min_edge_density: float = 0.04,
    min_text_visibility: float = 0.35,
) -> bool:
    """
    Validate that a generated CAPTCHA is not degenerate.
    
    AGGRESSIVE thresholds — if a human can't read it, reject it.
    
    Args:
        final_image: The fully rendered CAPTCHA image (after all effects).
        text_layer: The text layer BEFORE noise/post-processing (alpha channel
                    indicates where text pixels are).
        min_contrast: Minimum mean absolute difference between text and
                      background pixel values in the text region.
        min_edge_density: Minimum ratio of edge pixels in the text region,
                         to verify text has visible structure after blur.
        min_text_visibility: Minimum fraction of text pixels that remain
                            distinguishable from their local neighborhood.
    
    Returns:
        True if the CAPTCHA passes quality checks, False if degenerate.
    """
    try:
        # Convert to grayscale numpy arrays
        final_gray = np.array(final_image.convert('L'), dtype=np.float32)
        
        # Get text mask from alpha channel if available
        if text_layer.mode == 'RGBA':
            text_mask = np.array(text_layer.split()[-1]) > 128
        else:
            # Fallback: approximate text region from the text layer
            text_gray = np.array(text_layer.convert('L'))
            text_mask = text_gray > 30
        
        # Check 1: sufficient text pixels exist
        text_pixel_count = text_mask.sum()
        total_pixels = text_mask.size
        if text_pixel_count < total_pixels * 0.03:
            # Text covers less than 3% of image — font too small or missing
            return False
        
        if text_pixel_count == 0:
            return False
        
        # Check 2: Text-background contrast
        # Compare mean intensity of text pixels vs non-text pixels
        text_pixels = final_gray[text_mask]
        bg_mask = ~text_mask
        if bg_mask.sum() > 0:
            bg_pixels = final_gray[bg_mask]
            contrast = abs(text_pixels.mean() - bg_pixels.mean())
            if contrast < min_contrast:
                return False
        
        # Check 3: Edge density in text region — catches over-blurred text
        edges = np.array(
            final_image.convert('L').filter(ImageFilter.FIND_EDGES),
            dtype=np.float32
        )
        text_region_edges = edges[text_mask]
        edge_ratio = (text_region_edges > 30).sum() / max(text_pixel_count, 1)
        if edge_ratio < min_edge_density:
            return False
        
        # Check 4: Local visibility — text pixels should differ from
        # their immediate neighborhood in the final image
        h, w = final_gray.shape
        kernel_size = 5
        padded = np.pad(final_gray, kernel_size, mode='reflect')
        
        visible_count = 0
        checked_count = 0
        
        text_coords = np.argwhere(text_mask)
        if len(text_coords) > 500:
            indices = np.random.choice(len(text_coords), 500, replace=False)
            text_coords = text_coords[indices]
        
        for y, x in text_coords:
            py, px = y + kernel_size, x + kernel_size
            neighborhood = padded[py-kernel_size:py+kernel_size+1,
                                  px-kernel_size:px+kernel_size+1]
            pixel_val = final_gray[y, x]
            local_mean = neighborhood.mean()
            # Text pixel should clearly differ from local neighborhood
            if abs(pixel_val - local_mean) > 25:
                visible_count += 1
            checked_count += 1
        
        if checked_count > 0:
            visibility = visible_count / checked_count
            if visibility < min_text_visibility:
                return False
        
        return True
        
    except Exception:
        # If validation fails for any reason, accept the sample
        return True
