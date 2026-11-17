from PIL import Image, ImageFont, ImageDraw
import random

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 60
FONT_SIZE = 33
TEXT_HEIGHT_FRACTION = 0.55
TEXT_SPREAD_PCT = 80
CHAR_SPACING_PX_ESTIMATE = 25
BASE_COLOR_RGB = (240, 240, 240)
SECONDARY_COLORS_RGB = [
    (0, 200, 0),
    (255, 140, 0),
    (180, 180, 255),
    (200, 150, 200)
]
PRIMARY_COLORS_RGB = [
    (0, 200, 0),
    (255, 140, 0),
    (40, 40, 40),
    (0, 180, 0),
    (200, 150, 200)
]
NOISE_DENSITY = 0.8
TILE_SIZE_MIN = 5
TILE_SIZE_MAX = 15

SEARCH_SPACE = {
    "BASE_COLOR_RGB": [(240, 240, 240)],
    "SECONDARY_COLORS_RGB": [SECONDARY_COLORS_RGB],
    "PRIMARY_COLORS_RGB": [PRIMARY_COLORS_RGB],
    "NOISE_DENSITY": [0.8],
    "TILE_SIZE_MIN": [5],
    "TILE_SIZE_MAX": [15]
}

def get_font():
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except:
        return ImageFont.load_default()

def generate_tiled_background(width, height):
    img = Image.new("RGB", (width, height), BASE_COLOR_RGB)
    draw = ImageDraw.Draw(img)
    for y in range(0, height, random.randint(TILE_SIZE_MIN, TILE_SIZE_MAX)):
        for x in range(0, width, random.randint(TILE_SIZE_MIN, TILE_SIZE_MAX)):
            color = random.choice(SECONDARY_COLORS_RGB)
            draw.rectangle([x, y, x + random.randint(TILE_SIZE_MIN, TILE_SIZE_MAX), y + random.randint(TILE_SIZE_MIN, TILE_SIZE_MAX)], fill=color)
    return img

def generate_captcha(text: str):
    img = generate_tiled_background(IMAGE_WIDTH, IMAGE_HEIGHT)
    draw = ImageDraw.Draw(img)
    font = get_font()

    # Calculate total text width and spacing
    char_widths = [font.getbbox(char)[2] for char in text]
    total_text_width = sum(char_widths)
    total_spacing = IMAGE_WIDTH - total_text_width
    spacing = max(0, total_spacing // (len(text) - 1)) if len(text) > 1 else 0

    # Calculate starting x position to center text with spread
    start_x = (IMAGE_WIDTH - total_text_width - spacing * (len(text) - 1)) * (1 - TEXT_SPREAD_PCT / 100) / 2
    current_x = start_x

    # Draw each character
    for char in text:
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        draw.text((current_x, (IMAGE_HEIGHT - FONT_SIZE) // 2), char, font=font, fill=random.choice(PRIMARY_COLORS_RGB))
        current_x += char_width + spacing

    return img