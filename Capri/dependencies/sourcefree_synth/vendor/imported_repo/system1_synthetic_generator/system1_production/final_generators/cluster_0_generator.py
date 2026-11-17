from PIL import Image, ImageFont, ImageDraw
import random

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 60
FONT_SIZE = 30
TEXT_HEIGHT_FRACTION = 0.5
CHAR_SPACING = 10
NOISE_DENSITY = 0.2
BACKGROUND_COLOR = (255, 255, 255)
PRIMARY_COLORS = [
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (0, 0, 139),    # Dark blue
    (255, 255, 0)   # Yellow
]
NOISE_COLORS = [
    (220, 220, 220),
    (180, 180, 180)
]

SEARCH_SPACE = {
    "FONT_SIZE": (25, 35),
    "CHAR_SPACING": (8, 12),
    "NOISE_DENSITY": (0.1, 0.3),
    "PRIMARY_COLORS": [
        [(240, 0, 240), (255, 0, 255)],  # Magenta range
        [(0, 240, 240), (0, 255, 255)],  # Cyan range
        [(0, 0, 120), (0, 0, 150)],      # Dark blue range
        [(240, 240, 0), (255, 255, 0)]   # Yellow range
    ],
    "NOISE_COLORS": [
        [(200, 200, 200), (230, 230, 230)],
        [(160, 160, 160), (190, 190, 190)]
    ]
}

def get_font():
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except:
        return ImageFont.load_default()

def generate_noise(image, density, colors):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(int(density * width * height)):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        draw.point((x, y), fill=random.choice(colors))

def draw_text_with_spacing(draw, font, text, colors, x, y):
    total_width = 0
    for char in text:
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        total_width += char_width + CHAR_SPACING
    start_x = (IMAGE_WIDTH - total_width) // 2
    for char in text:
        color = random.choice(colors)
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        draw.text((start_x, y), char, font=font, fill=color)
        start_x += char_width + CHAR_SPACING

def generate_captcha(text: str):
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), BACKGROUND_COLOR)
    generate_noise(image, NOISE_DENSITY, NOISE_COLORS)
    font = get_font()
    draw = ImageDraw.Draw(image)
    y = (IMAGE_HEIGHT - FONT_SIZE) // 2
    draw_text_with_spacing(draw, font, text, PRIMARY_COLORS, 0, y)
    return image