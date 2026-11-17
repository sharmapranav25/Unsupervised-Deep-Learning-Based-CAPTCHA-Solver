from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 60
FONT_SIZE = 39
TEXT_HEIGHT_FRACTION = 0.65
TEXT_WEIGHT = "regular"
TEXT_STYLE = "sans-serif"
SHADOW_OFFSET = (1, 1)
PRIMARY_COLORS = [
    (10, 10, 10),
    (200, 100, 50),
    (0, 200, 100),
    (255, 0, 255),
    (0, 0, 100)
]
TEXT_SPREAD_PCT = 70
CHAR_COUNT_TYPICAL = 5
CHAR_SPACING_ESTIMATE = 15
NOISE_DENSITY = 0.1
NOISE_COLOR = (0, 0, 0)

SEARCH_SPACE = {
    "TEXT_HEIGHT_FRACTION": (0.5, 0.7),
    "TEXT_SPREAD_PCT": (60, 80),
    "NOISE_DENSITY": (0.05, 0.15),
    "CHAR_SPACING_ESTIMATE": (10, 20)
}

def get_font():
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except:
        return ImageFont.load_default()

def add_noise(image, density, color):
    pixels = image.load()
    width, height = image.size
    for _ in range(int(density * width * height)):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        pixels[x, y] = color
    return image

def generate_captcha(text: str):
    font = get_font()
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    image = add_noise(image, NOISE_DENSITY, NOISE_COLOR)

    char_widths = [font.getbbox(char)[2] for char in text]
    total_width = sum(char_widths)
    spacing = max(1, (IMAGE_WIDTH - total_width) // (len(text) - 1) if len(text) > 1 else 0)

    x_offset = (IMAGE_WIDTH - total_width - spacing * (len(text) - 1)) // 2
    if x_offset < 0:
        x_offset = 0

    for i, char in enumerate(text):
        color = random.choice(PRIMARY_COLORS)
        shadow_color = tuple(max(0, c - 50) for c in color)

        shadow_x = x_offset + i * (char_widths[i] + spacing) + SHADOW_OFFSET[0]
        shadow_y = (IMAGE_HEIGHT - FONT_SIZE) // 2 + SHADOW_OFFSET[1]
        draw.text((shadow_x, shadow_y), char, font=font, fill=shadow_color)

        text_x = x_offset + i * (char_widths[i] + spacing)
        text_y = (IMAGE_HEIGHT - FONT_SIZE) // 2
        draw.text((text_x, text_y), char, font=font, fill=color)

    return image