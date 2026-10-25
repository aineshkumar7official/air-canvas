# distance between two points
import math
#decodes pictures embedded in stylus SVG icons
import base64
#finds the fill colors inside stylus SVG icons
import re
# standard libraries for time and URL handling
import time
#names emoji so the Trace & Play picker can search them
import unicodedata
#downloads hand model file
import urllib.request
#create screenshot info and saves screenshot
from datetime import datetime
from pathlib import Path

#opencv for computer vision tasks
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageDraw, ImageFont

#renders the SVG stylus icons; the menu falls back to stroke previews without it
try:
    import pymupdf
except ImportError:
    pymupdf = None


## creates frame for project
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "Air Canvas"

#design assets directories and model URL

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_NAME = "hand_landmarker.task"
DOWNLOADS_DIR = Path.home() / "Downloads"
ASSETS_DIR_NAME = "AirCanvas Interface"
FONTS_DIR_NAME = "AirCanvas Fonts"
STYLUS_ICONS_DIR_NAME = "Air Canvas Stylus Icons"

#size and spacing of buttons. 
TOOLBAR_TOP = 10
TOOLBAR_HEIGHT = 80
BUTTON_WIDTH = 110
BUTTON_GAP = 15
TOOLBAR_LEFT = 20


#==============================================================================
# SHARED HELPERS
# used by every section below: assets, hit-testing, blending, fonts, icons
#==============================================================================


#gets assets directory
def find_assets_dir(name, fallback):
    candidates = [Path(__file__).resolve().parent / name, DOWNLOADS_DIR / name]
    for path in candidates:
        if path.is_dir():
            return path
    print(f"'{name}' folder not found - using {fallback}")
    return None


#downloads or finds the hand model file
def find_or_download_model():
    candidates = [Path(__file__).resolve().parent / MODEL_NAME, DOWNLOADS_DIR / MODEL_NAME]
    for path in candidates:
        if path.exists():
            return path
    target = candidates[0]
    print(f"Downloading hand model to {target} ...")
    urllib.request.urlretrieve(MODEL_URL, target)
    return target

#checks if a point is inside a given box
def point_in_box(point, box):
    if point is None:
        return False
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


#pastes image onto the frame with alpha blending
def blend(frame, image, left, top):
    frame_height, frame_width = frame.shape[:2]
    x1, y1 = max(left, 0), max(top, 0)
    x2, y2 = min(left + image.shape[1], frame_width), min(top + image.shape[0], frame_height)
    if x1 >= x2 or y1 >= y2:
        return
    patch = image[y1 - top:y2 - top, x1 - left:x2 - left]
    region = frame[y1:y2, x1:x2]
    alpha = patch[:, :, 3:] / 255.0
    region[:] = (patch[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)


#font management class
class Typography:
    FILES = {
        "display": "loubag-regular/loubag-regular.ttf",
        "regular": "Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf",
        "semibold": "Plus_Jakarta_Sans/static/PlusJakartaSans-SemiBold.ttf",
        "bold": "Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf",
        "italic": "Plus_Jakarta_Sans/static/PlusJakartaSans-BoldItalic.ttf",
    }
    CACHE_LIMIT = 256

    def __init__(self):
        self.folder = find_assets_dir(FONTS_DIR_NAME, "the default font")
        self.fonts = {}
        self.tiles = {}

    def font(self, style, size):
        key = (style, size)
        if key not in self.fonts:
            try:
                self.fonts[key] = ImageFont.truetype(str(self.folder / self.FILES[style]), size)
            except (OSError, TypeError):
                self.fonts[key] = ImageFont.load_default(size)
        return self.fonts[key]

    def measure(self, text, style, size):
        x1, y1, x2, y2 = self.font(style, size).getbbox(text)
        return x2 - x1, y2 - y1

    def tile(self, text, style, size, color, stroke):
        key = (text, style, size, color, stroke)
        if key not in self.tiles:
            if len(self.tiles) >= self.CACHE_LIMIT:
                self.tiles.clear()
            font = self.font(style, size)
            x1, y1, x2, y2 = font.getbbox(text, stroke_width=stroke)
            image = Image.new("RGBA", (max(1, x2 - x1), max(1, y2 - y1)), (0, 0, 0, 0))
            # colors stay in BGR order, so the RGBA array comes out as BGRA for OpenCV
            ImageDraw.Draw(image).text((-x1, -y1), text, font=font, fill=tuple(color), stroke_width=stroke, stroke_fill=(0, 0, 0))
            self.tiles[key] = np.array(image)
        return self.tiles[key]

    def draw(self, frame, text, position, style, size, color=(255, 255, 255), stroke=0, centered=False):
        tile = self.tile(text, style, size, color, stroke)
        x, y = position
        if centered:
            x -= tile.shape[1] // 2
            y -= tile.shape[0] // 2
        blend(frame, tile, x, y)


TEXT = Typography()

#drawing supporters
def draw_label(frame, text, box, text_color=(255, 255, 255)):
    x1, y1, x2, y2 = box
    TEXT.draw(frame, text, ((x1 + x2) // 2, (y1 + y2) // 2), "display", 22, text_color, stroke=2, centered=True)


def draw_selection(frame, box, color=(255, 255, 255)):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), color, 4)


#UI elements
class IconSet:
    FILES = {
        "Red": "1.png",
        "Orange": "2.png",
        "Yellow": "3.png",
        "Green": "4.png",
        "Blue": "5.png",
        "Purple": "6.png",
        "eraser": "7.png",
        "camera": "8.png",
        "draw": "9.png",
        "cursor": "10.png",
        "fist": "11.png",
        "info": "Colorful Think Outside the Box Flashcards.png",
    }
    TINT_SIZE = 400

    #forms the transparent background
    def __init__(self):
        folder = find_assets_dir(ASSETS_DIR_NAME, "built-in drawings")
        self.images = {}
        self.cache = {}
        if folder is None:
            return
        for name, filename in self.FILES.items():
            image = cv2.imread(str(folder / filename), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3 or image.shape[2] != 4:
                print(f"Skipping {filename}: expected a transparent PNG")
                continue
            if image[:, :, 3].min() > 0:
                image = self.remove_background(image)
            ys, xs = np.nonzero(image[:, :, 3])
            self.images[name] = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    @staticmethod
    #makes the white background transparent
    def remove_background(image):
        # opaque artwork gets its light background (the part touching the edges) made transparent
        light = np.all(image[:, :, :3] > 230, axis=2).astype(np.uint8)
        _, labels = cv2.connectedComponents(light)
        edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        image = image.copy()
        image[np.isin(labels, edge_labels[edge_labels > 0]), 3] = 0
        return image

    #checks if icon loads
    def has(self, name):
        return name in self.images

    #scales the icon to fit within the specified width and height
    def fitted(self, name, width, height):
        key = (name, width, height)
        if key not in self.cache:
            image = self.images[name]
            scale = min(width / image.shape[1], height / image.shape[0])
            size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
            self.cache[key] = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return self.cache[key]

    #a copy of the icon with its fill recolored, made the first time each color is asked for
    def tinted(self, name, color):
        key = (name, color)
        if key not in self.images:
            # shrunk first so recoloring the full-size artwork doesn't stall a frame
            image = self.images[name]
            scale = min(self.TINT_SIZE / max(image.shape[:2]), 1)
            small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            self.images[key] = tint_fill(small, color)
        return key

    #draws the icon on the frame at the specified box location
    def draw(self, frame, name, box):
        x1, y1, x2, y2 = box
        icon = self.fitted(name, x2 - x1, y2 - y1)
        height, width = icon.shape[:2]
        blend(frame, icon, x1 + (x2 - x1 - width) // 2, y1 + (y2 - y1 - height) // 2)


#swaps the red accent shades in a list of BGR colors for shades of the target color
def tint_accent(colors, target):
    hsv = cv2.cvtColor(colors.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    # strong reds only, so pink erasers, tan wood, gray ferrules and black outlines keep their color
    accent = ((hsv[:, 0] <= 10) | (hsv[:, 0] >= 170)) & (hsv[:, 1] >= 128) & (hsv[:, 2] >= 25)
    if not accent.any():
        return colors, accent
    goal = cv2.cvtColor(np.uint8([[target]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    # the brightest red becomes the target color and darker reds become darker shades of it
    brightest = np.percentile(hsv[accent, 2], 95)
    shades = hsv[accent]
    shades[:, 0], shades[:, 1] = goal[0], goal[1]
    shades[:, 2] = np.clip(goal[2] * shades[:, 2] / brightest, 0, 255)
    result = colors.copy()
    result[accent] = cv2.cvtColor(shades.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return result, accent


#repaints a color button's fill in the target color, keeping its outline and label
def tint_fill(image, target):
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    # the fill is the only saturated part; outlines and labels are black or white
    fill = hsv[:, :, 1] >= 80
    if not fill.any():
        return image
    goal = cv2.cvtColor(np.uint8([[target]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    brightest = np.percentile(hsv[fill][:, 2], 95)
    shades = hsv[fill]
    shades[:, 0], shades[:, 1] = goal[0], goal[1]
    shades[:, 2] = np.clip(goal[2] * shades[:, 2] / brightest, 0, 255)
    result = image.copy()
    result[fill, :3] = cv2.cvtColor(shades.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    # white labels turn dark on light shades so they stay readable
    blue, green, red = target
    if 0.114 * blue + 0.587 * green + 0.299 * red > 170:
        label = ~fill & (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 128)
        result[label, :3] = 255 - result[label, :3]
    return result


#stylus icons drawn from SVG and recolored to match the selected stroke color
class StylusIconSet(IconSet):
    FILES = {
        "Pencil": "13.svg",
        "Pen": "14.svg",
        "Highlighter": "15.svg",
        "Paint Brush": "16.svg",
        "Crayon": "17.svg",
    }
    FILL = re.compile(r'fill="#([0-9a-fA-F]{6})"')
    PICTURE = re.compile(r'base64,([A-Za-z0-9+/=]+)')
    PICTURE_SIZE = 800

    #reads the SVG sources; icons are rendered per color the first time they are drawn
    def __init__(self):
        self.sources = {}
        self.images = {}
        self.cache = {}
        folder = find_assets_dir(STYLUS_ICONS_DIR_NAME, "stroke previews")
        if folder is None:
            return
        if pymupdf is None:
            print("pymupdf not installed - using stroke previews")
            return
        for name, filename in self.FILES.items():
            path = folder / filename
            if path.exists():
                self.sources[name] = path.read_text()
            else:
                print(f"Skipping {filename}: not found")

    def has(self, name):
        return name in self.sources

    @staticmethod
    def render(svg):
        page = pymupdf.open(stream=svg.encode(), filetype="svg")[0]
        pixmap = page.get_pixmap(alpha=True)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 4)
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

    #rebuilds a picture-only SVG from its embedded layers, since pymupdf skips SVG masks
    @classmethod
    def picture(cls, svg):
        layers = [
            cv2.imdecode(np.frombuffer(base64.b64decode(data), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            for data in cls.PICTURE.findall(svg)
        ]
        art = next((layer for layer in layers if layer.ndim == 3), None)
        if art is None:
            return cls.render(svg)
        image = cv2.cvtColor(art, cv2.COLOR_BGR2BGRA) if art.shape[2] == 3 else art.copy()
        # Canva saves the picture's transparency as a separate grayscale mask layer
        mask = next((layer for layer in layers if layer.ndim == 2 and layer.shape == art.shape[:2]), None)
        if mask is not None:
            image[:, :, 3] = mask
        scale = min(cls.PICTURE_SIZE / max(image.shape[:2]), 1)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    #recolors the vector fills when the SVG has them, otherwise the rendered pixels
    def load(self, name, color):
        svg = self.sources[name]
        fills = sorted({fill.lower() for fill in self.FILL.findall(svg)})
        bgr = np.array([[int(f[4:6], 16), int(f[2:4], 16), int(f[0:2], 16)] for f in fills], dtype=np.uint8).reshape(-1, 3)
        tinted, accent = tint_accent(bgr, color)
        vector = accent.any()
        if vector:
            swaps = {f: "%02x%02x%02x" % (r, g, b) for f, (b, g, r) in zip(fills, tinted)}
            svg = self.FILL.sub(lambda m: f'fill="#{swaps[m.group(1).lower()]}"', svg)
        image = self.render(svg) if vector else self.picture(svg)
        if image[:, :, 3].min() > 0:
            image = self.remove_background(image)
        if not vector:
            # artwork exported as an embedded picture has no fills to swap
            pixels = image[:, :, :3].reshape(-1, 3)
            image[:, :, :3] = tint_accent(pixels, color)[0].reshape(image.shape[0], image.shape[1], 3)
        ys, xs = np.nonzero(image[:, :, 3])
        return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    #icons are keyed by (stylus name, color) so each stroke color gets its own tinted copy
    def fitted(self, key, width, height):
        if key not in self.images:
            self.images[key] = self.load(*key)
        return super().fitted(key, width, height)


#camera icon location
def draw_camera_icon(frame, box, body_color):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    body_top = y1 + int(height * 0.19)
    body_height = y2 - body_top
    cv2.rectangle(frame, (x1, body_top), (x2, y2), body_color, cv2.FILLED)
    cv2.rectangle(frame, (x1 + int(width * 0.29), y1), (x1 + int(width * 0.62), body_top), body_color, cv2.FILLED)
    center = ((x1 + x2) // 2, (body_top + y2) // 2)
    cv2.circle(frame, center, int(body_height * 0.37), (255, 255, 255), 3)
    cv2.circle(frame, center, int(body_height * 0.18), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(frame, (x1, body_top), (x2, y2), (255, 255, 255), 2)

#hand icon location
def draw_hand_icon(frame, center, raised, highlight=(0, 220, 255)):
    cx, cy = center
    skin = (200, 210, 225)
    outline = (40, 40, 40)
    palm_top = cy - 35
    finger_x = [cx - 27, cx - 9, cx + 9, cx + 27]
    finger_lengths = [62, 70, 64, 52]
    for x, length, up in zip(finger_x, finger_lengths, raised):
        top = palm_top - length if up else palm_top - 12
        color = highlight if up else skin
        cv2.rectangle(frame, (x - 7, top + 7), (x + 7, palm_top + 10), color, cv2.FILLED)
        cv2.circle(frame, (x, top + 7), 7, color, cv2.FILLED)
        cv2.rectangle(frame, (x - 7, top + 7), (x + 7, palm_top + 10), outline, 1)
    cv2.rectangle(frame, (cx - 36, palm_top), (cx + 36, cy + 35), skin, cv2.FILLED)
    cv2.rectangle(frame, (cx - 36, palm_top), (cx + 36, cy + 35), outline, 2)
    cv2.ellipse(frame, (cx - 30, cy + 5), (12, 24), 30, 0, 360, skin, cv2.FILLED)
    cv2.ellipse(frame, (cx - 30, cy + 5), (12, 24), 30, 0, 360, outline, 2)
    cv2.rectangle(frame, (cx - 30, cy + 35), (cx + 30, cy + 55), skin, cv2.FILLED)
    cv2.rectangle(frame, (cx - 30, cy + 35), (cx + 30, cy + 55), outline, 2)

#centered text drawing function
def draw_centered(frame, text, center_y, style, size, color=(255, 255, 255), center_x=None, stroke=0):
    x = frame.shape[1] // 2 if center_x is None else center_x
    TEXT.draw(frame, text, (x, center_y), style, size, color, stroke, centered=True)

#wraps text to fit within a specified width
def wrap_text(text, max_width, style, size):
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if TEXT.measure(candidate, style, size)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


#==============================================================================
# 1. CANVAS
# where drawing is allowed, the no-draw zones, edge warnings, and the drawing itself
#==============================================================================


#drawing area for creating artwork
class DrawingZone:
    BORDER_THICKNESS = 6
    TOP_GAP = 60
    BOTTOM_GAP = 90

    def __init__(self, frame_width, frame_height):
        self.box = (20, TOOLBAR_TOP + TOOLBAR_HEIGHT + self.TOP_GAP, frame_width - 20, frame_height - self.BOTTOM_GAP)

    def contains(self, point):
        x1, y1, x2, y2 = self.box
        inset = self.BORDER_THICKNESS
        return point_in_box(point, (x1 + inset, y1 + inset, x2 - inset, y2 - inset))

    def clip(self, canvas):
        x1, y1, x2, y2 = self.box
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        canvas[mask == 0] = 0

    def draw(self, frame, color=(0, 0, 0)):
        x1, y1, x2, y2 = self.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.BORDER_THICKNESS)


#drawing motion
class DrawingCanvas:
    BRUSH_THICKNESS = 8

    def __init__(self, width, height):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.previous_point = None

    def draw_line(self, point, color):
        if self.previous_point is None:
            self.previous_point = point
        cv2.line(self.canvas, self.previous_point, point, color, self.BRUSH_THICKNESS)
        self.previous_point = point

    def lift_pen(self):
        self.previous_point = None

    def clear(self):
        self.canvas[:] = 0
        self.lift_pen()

    def overlay(self, frame):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inverse_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        inverse_mask = cv2.cvtColor(inverse_mask, cv2.COLOR_GRAY2BGR)
        background = cv2.bitwise_and(frame, inverse_mask)
        return cv2.bitwise_or(background, self.canvas)


#defines safezones areas
class SafeZones:
    CORNER_SIZE = 200
    TINT = (0, 0, 200)
    TINT_ALPHA = 0.3
    LABEL_COLOR = (220, 220, 220)

    def __init__(self, frame_width, frame_height, drawing_zone):
        zone_x1, zone_y1, zone_x2, zone_y2 = drawing_zone.box
        self.drawing_box = drawing_zone.box
        self.top_bar = (0, 0, frame_width, zone_y1)
        self.bottom_bar = (0, zone_y2, frame_width, frame_height)
        self.left_margin = (0, zone_y1, zone_x1, zone_y2)
        self.right_margin = (zone_x2, zone_y1, frame_width, zone_y2)
        self.camera_corner = (frame_width - self.CORNER_SIZE, 0, frame_width, self.CORNER_SIZE)
        self.areas = {
            "top menu": self.top_bar,
            "camera corner": self.camera_corner,
            "bottom interface": self.bottom_bar,
            "left margin": self.left_margin,
            "right margin": self.right_margin,
        }
        # the camera corner overlaps the top bar and right margin, so only its in-canvas part is shaded on its own
        corner_x1, _, _, corner_y2 = self.camera_corner
        self.shaded = [
            self.top_bar,
            self.bottom_bar,
            self.left_margin,
            self.right_margin,
            (corner_x1, zone_y1, zone_x2, corner_y2),
        ]

    def blocks(self, point):
        return any(point_in_box(point, box) for box in self.areas.values())

    def clip(self, canvas):
        for x1, y1, x2, y2 in self.areas.values():
            canvas[y1:y2, x1:x2] = 0

    def shade(self, frame, box):
        x1, y1, x2, y2 = box
        region = frame[y1:y2, x1:x2]
        tint = np.empty_like(region)
        tint[:] = self.TINT
        region[:] = cv2.addWeighted(region, 1 - self.TINT_ALPHA, tint, self.TINT_ALPHA, 0)

    def draw_label(self, frame, label, label_top):
        TEXT.draw(frame, label, (self.drawing_box[0] + 12, label_top), "semibold", 14, self.LABEL_COLOR)

    def draw(self, frame):
        for box in self.shaded:
            self.shade(frame, box)
        self.draw_label(frame, "MENU ZONE - NO DRAWING", self.top_bar[3] - 24)
        self.draw_label(frame, "INTERFACE ZONE - NO DRAWING", self.bottom_bar[1] + 9)


#warning system for approaching edges and restricted areas
class EdgeWarning:
    BUFFER = 45
    FLASH_HZ = 4
    WARNING_COLOR = (0, 0, 255)
    BANNER_COLOR = (0, 0, 160)
    BANNER_ALPHA = 0.7
    EDGE = "edge"
    RESTRICTED = "restricted"
    MESSAGES = {
        EDGE: "WARNING: Approaching Edge",
        RESTRICTED: "Restricted Area - Drawing Disabled",
    }

    def __init__(self, drawing_zone, safe_zones):
        x1, _, x2, _ = drawing_zone.box
        top_line = safe_zones.top_bar[3]
        bottom_line = safe_zones.bottom_bar[1]
        self.safe_zones = safe_zones
        self.safe_box = (x1 + self.BUFFER, top_line + self.BUFFER, x2 - self.BUFFER, bottom_line - self.BUFFER)
        cx1, cy1, cx2, cy2 = safe_zones.camera_corner
        self.camera_buffer = (cx1 - self.BUFFER, cy1, cx2, cy2 + self.BUFFER)
        self.banner_y = (TOOLBAR_TOP + TOOLBAR_HEIGHT + top_line) // 2
        self.state = None

    @property
    def restricted(self):
        return self.state == self.RESTRICTED

    def update(self, gesture, point, on_button):
        self.state = None
        if gesture != "draw" or point is None or on_button:
            return
        if self.safe_zones.blocks(point):
            self.state = self.RESTRICTED
        elif not point_in_box(point, self.safe_box) or point_in_box(point, self.camera_buffer):
            self.state = self.EDGE

    def border_color(self):
        flash_on = int(time.monotonic() * self.FLASH_HZ * 2) % 2 == 0
        if self.restricted or flash_on:
            return self.WARNING_COLOR
        return (0, 0, 0)

    def draw_banner(self, frame):
        text = self.MESSAGES[self.state]
        width, height = TEXT.measure(text, "bold", 20)
        center_x = frame.shape[1] // 2
        x1, y1 = center_x - width // 2 - 18, self.banner_y - height // 2 - 6
        x2, y2 = center_x + width // 2 + 18, self.banner_y + height // 2 + 6
        region = frame[y1:y2, x1:x2]
        overlay = region.copy()
        overlay[:] = self.BANNER_COLOR
        TEXT.draw(overlay, text, ((x2 - x1) // 2, (y2 - y1) // 2), "bold", 20, centered=True)
        region[:] = cv2.addWeighted(overlay, self.BANNER_ALPHA, region, 1 - self.BANNER_ALPHA, 0)

    def draw(self, frame, drawing_zone):
        if self.state is None:
            return
        drawing_zone.draw(frame, self.border_color())
        self.draw_banner(frame)


#==============================================================================
# 2. TOOLS
# toolbar buttons: colors, hover-to-select, eraser, screenshot, info, status bar
#==============================================================================


#color palette for selecting drawing colors; hovering a color also drops down its shades
class ColorPalette:
    SHADES = {
        "Red": [("Light Red", (102, 102, 255)), ("Crimson", (60, 20, 220)), ("Maroon", (0, 0, 128))],
        "Orange": [("Peach", (150, 200, 255)), ("Coral", (80, 127, 255)), ("Burnt Orange", (0, 85, 204))],
        "Yellow": [("Lemon", (153, 255, 255)), ("Gold", (0, 215, 255)), ("Mustard", (1, 173, 225))],
        "Green": [("Lime", (50, 205, 50)), ("Mint", (152, 251, 152)), ("Olive", (0, 128, 128))],
        "Blue": [("Sky", (235, 206, 135)), ("Royal", (225, 105, 65)), ("Navy", (128, 0, 0))],
        "Purple": [("Lavender", (220, 126, 181)), ("Violet", (211, 0, 148)), ("Plum", (102, 51, 102))],
    }
    SHADE_HEIGHT = 40
    SHADE_GAP = 6
    DROP_GAP = 8
    PANEL_PADDING = 6
    PANEL_SHADE = 30
    PANEL_ALPHA = 0.75
    CHECK_RADIUS = 10
    CLOSE_DELAY = 0.8
    INK = (40, 40, 40)
    OPEN_COLOR = (0, 220, 255)

    def __init__(self, icons):
        self.icons = icons
        self.colors = [
            ("Red", (0, 0, 255)),
            ("Orange", (0, 165, 255)),
            ("Yellow", (0, 255, 255)),
            ("Green", (0, 128, 0)),
            ("Blue", (255, 0, 0)),
            ("Purple", (128, 0, 128)),
        ]
        self.boxes = []
        self.dropdowns = {}
        for i, (name, color) in enumerate(self.colors):
            x1 = TOOLBAR_LEFT + i * (BUTTON_WIDTH + BUTTON_GAP)
            box = (x1, TOOLBAR_TOP, x1 + BUTTON_WIDTH, TOOLBAR_TOP + TOOLBAR_HEIGHT)
            self.boxes.append((name, color, box))
            # the base color leads its own dropdown so it can always be picked again
            self.dropdowns[name] = self.layout_dropdown(box, [(name, color)] + self.SHADES[name])
        self.selected_name, self.selected_color = self.colors[0]
        self.selected_family = self.selected_name
        self.open_name = None
        self.away_since = None

    #stacks the shade swatches in a column under the color button
    def layout_dropdown(self, button, shades):
        x1, _, x2, y2 = button
        top = y2 + self.DROP_GAP
        options = []
        for i, (name, color) in enumerate(shades):
            y1 = top + i * (self.SHADE_HEIGHT + self.SHADE_GAP)
            options.append((name, color, (x1, y1, x2, y1 + self.SHADE_HEIGHT)))
        pad = self.PANEL_PADDING
        bottom = options[-1][2][3]
        return options, (x1 - pad, top - pad, x2 + pad, bottom + pad)

    @property
    def right_edge(self):
        return self.boxes[-1][2][2]

    @property
    def is_open(self):
        return self.open_name is not None

    def box_at(self, point):
        for name, color, box in self.boxes:
            if point_in_box(point, box):
                return name, box
        return None

    def contains(self, point):
        if self.box_at(point) is not None:
            return True
        return self.is_open and point_in_box(point, self.dropdowns[self.open_name][1])

    def target_at(self, point):
        if self.is_open:
            for name, color, box in self.dropdowns[self.open_name][0]:
                if point_in_box(point, box):
                    return ("shade", name), box
        hit = self.box_at(point)
        if hit:
            name, box = hit
            return ("color", name), box
        return None

    def open(self, name):
        self.open_name = name
        self.away_since = None

    def close(self):
        self.open_name = None
        self.away_since = None

    def select(self, name):
        for color_name, color, box in self.boxes:
            if color_name == name:
                self.selected_name, self.selected_color = color_name, color
                self.selected_family = color_name

    def select_shade(self, name):
        for family, (options, panel) in self.dropdowns.items():
            for shade_name, color, box in options:
                if shade_name == name:
                    self.selected_name, self.selected_color = shade_name, color
                    self.selected_family = family
        self.close()

    #closes the dropdown once the cursor has been away from it for a moment
    def update(self, gesture, point):
        if not self.is_open:
            return
        if gesture == "cursor" and self.contains(point):
            self.away_since = None
            return
        now = time.monotonic()
        if self.away_since is None:
            self.away_since = now
        elif now - self.away_since >= self.CLOSE_DELAY:
            self.close()

    def draw_dropdown(self, frame):
        options, panel = self.dropdowns[self.open_name]
        x1, y1, x2, y2 = panel
        region = frame[y1:y2, x1:x2]
        shade = np.full_like(region, self.PANEL_SHADE)
        region[:] = cv2.addWeighted(region, 1 - self.PANEL_ALPHA, shade, self.PANEL_ALPHA, 0)
        for name, color, box in options:
            bx1, by1, bx2, by2 = box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, cv2.FILLED)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), self.INK, 2)
            TEXT.draw(frame, name, ((bx1 + bx2) // 2, (by1 + by2) // 2), "display", 15, stroke=2, centered=True)
            if name == self.selected_name:
                draw_selection(frame, box)
                self.draw_check(frame, box)

    #check badge pinned to the top-right corner of the active shade
    def draw_check(self, frame, box):
        _, y1, x2, _ = box
        cx, cy = x2 - 2, y1 + 2
        r = self.CHECK_RADIUS
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, self.INK, 2, cv2.LINE_AA)
        tick = np.array([(cx - r // 2, cy), (cx - r // 6, cy + r // 3), (cx + r // 2, cy - r // 3)], dtype=np.int32)
        cv2.polylines(frame, [tick], False, self.INK, 2, cv2.LINE_AA)

    def draw(self, frame, eraser_active):
        for name, color, box in self.boxes:
            x1, y1, x2, y2 = box
            highlighted = name == self.selected_family and not eraser_active
            # the button of the active family takes on the chosen shade
            shade = self.selected_color if name == self.selected_family else color
            if self.icons.has(name):
                icon = name if shade == color else self.icons.tinted(name, shade)
                self.icons.draw(frame, icon, box)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), shade, cv2.FILLED)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), 2)
                draw_label(frame, name, box)
            if name == self.open_name:
                draw_selection(frame, box, self.OPEN_COLOR)
            elif highlighted:
                draw_selection(frame, box)
        if self.is_open:
            self.draw_dropdown(frame)


#hover selector for UI elements
class HoverSelector:
    DWELL_SECONDS = 0.6
    PULSE_SECONDS = 0.4
    RING_RADIUS = 20

    def __init__(self):
        self.target = None
        self.target_box = None
        self.hover_start = None
        self.confirmed = False
        self.pulse_box = None
        self.pulse_start = 0

    def reset(self):
        self.target = None
        self.target_box = None
        self.hover_start = None
        self.confirmed = False

    def update(self, target, box):
        if target is None:
            self.reset()
            return None
        now = time.monotonic()
        if target != self.target:
            self.target, self.target_box = target, box
            self.hover_start = now
            self.confirmed = False
        if not self.confirmed and now - self.hover_start >= self.DWELL_SECONDS:
            self.confirmed = True
            self.pulse_box = box
            self.pulse_start = now
            return target
        return None

    def progress(self):
        if self.hover_start is None:
            return 0.0
        return min((time.monotonic() - self.hover_start) / self.DWELL_SECONDS, 1.0)

    def draw_progress(self, frame, point):
        if self.target is None or self.confirmed or point is None:
            return
        progress = self.progress()
        x1, y1, x2, y2 = self.target_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.rectangle(frame, (x1, y2 - 8), (x1 + int((x2 - x1) * progress), y2), (255, 255, 255), cv2.FILLED)
        cv2.ellipse(frame, point, (self.RING_RADIUS, self.RING_RADIUS), -90, 0, int(360 * progress), (255, 255, 255), 4)

    def draw_pulse(self, frame):
        elapsed = (time.monotonic() - self.pulse_start) / self.PULSE_SECONDS
        if self.pulse_box is None or elapsed >= 1:
            return
        grow = int(4 + 16 * elapsed)
        thickness = max(1, int(8 * (1 - elapsed)))
        x1, y1, x2, y2 = self.pulse_box
        cv2.rectangle(frame, (x1 - grow, y1 - grow), (x2 + grow, y2 + grow), (255, 255, 255), thickness)

    def draw(self, frame, point):
        self.draw_progress(frame, point)
        self.draw_pulse(frame)


#erasing function for removing parts of the drawing
class Eraser:
    RADIUS = 25

    def __init__(self, left_x, icons):
        self.icons = icons
        self.box = (left_x, TOOLBAR_TOP, left_x + BUTTON_WIDTH, TOOLBAR_TOP + TOOLBAR_HEIGHT)
        self.active = False
        self.previous_point = None

    def contains(self, point):
        return point_in_box(point, self.box)

    def erase(self, canvas, point):
        if self.previous_point is None:
            self.previous_point = point
        cv2.line(canvas.canvas, self.previous_point, point, (0, 0, 0), self.RADIUS * 2)
        cv2.circle(canvas.canvas, point, self.RADIUS, (0, 0, 0), cv2.FILLED)
        self.previous_point = point

    def lift(self):
        self.previous_point = None

    def draw(self, frame):
        x1, y1, x2, y2 = self.box
        if self.icons.has("eraser"):
            self.icons.draw(frame, "eraser", self.box)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), cv2.FILLED)
            cv2.rectangle(frame, (x1 + 12, y1 + 25), (x1 + 32, y2 - 25), (180, 130, 255), cv2.FILLED)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), 2)
            draw_label(frame, "Eraser", (x1 + 36, y1, x2, y2))
        if self.active:
            draw_selection(frame, self.box)

    def draw_cursor(self, frame, point):
        cv2.circle(frame, point, self.RADIUS, (0, 0, 0), 4)
        cv2.circle(frame, point, self.RADIUS, (255, 255, 255), 2)
        cv2.circle(frame, point, 3, (255, 255, 255), cv2.FILLED)


#screenshot function 
class ScreenshotCamera:
    ICON_WIDTH = 120
    ICON_HEIGHT = TOOLBAR_HEIGHT
    MARGIN = 20
    COUNTDOWN_SECONDS = 5

    def __init__(self, frame_width, save_dir, icons):
        self.icons = icons
        x2 = frame_width - self.MARGIN
        self.box = (x2 - self.ICON_WIDTH, TOOLBAR_TOP, x2, TOOLBAR_TOP + self.ICON_HEIGHT)
        self.save_dir = Path(save_dir)
        self.was_pressed = False
        self.countdown_start = None
        self.flash_until = 0
        self.last_saved = None

    def contains(self, point):
        return point_in_box(point, self.box)

    @property
    def counting_down(self):
        return self.countdown_start is not None

    def update_press(self, fingertip):
        pressed = self.contains(fingertip)
        if pressed and not self.was_pressed and not self.counting_down:
            self.countdown_start = time.monotonic()
        self.was_pressed = pressed

    def seconds_left(self):
        return self.COUNTDOWN_SECONDS - (time.monotonic() - self.countdown_start)

    def capture_if_ready(self, image):
        if not self.counting_down or self.seconds_left() > 0:
            return None
        self.save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"airCanvas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = self.save_dir / filename
        cv2.imwrite(str(path), image)
        self.countdown_start = None
        self.flash_until = time.monotonic() + 0.3
        self.last_saved = (path, time.monotonic())
        return path

    def draw_icon(self, frame):
        active = self.was_pressed or self.counting_down
        if self.icons.has("camera"):
            self.icons.draw(frame, "camera", self.box)
            if active:
                draw_selection(frame, self.box, (0, 200, 255))
        else:
            draw_camera_icon(frame, self.box, (0, 200, 255) if active else (60, 60, 60))

    def draw_countdown(self, frame):
        if not self.counting_down:
            return
        number = str(max(math.ceil(self.seconds_left()), 1))
        draw_centered(frame, number, frame.shape[0] // 2, "display", 240, stroke=6)

    def draw_feedback(self, frame):
        if time.monotonic() < self.flash_until:
            frame[:] = 255
        if self.last_saved and time.monotonic() - self.last_saved[1] < 3:
            message = f"Saved: {self.last_saved[0].name}"
            TEXT.draw(frame, message, (700, frame.shape[0] - 38), "semibold", 20, (0, 255, 0), stroke=2)

    def draw(self, frame):
        self.draw_icon(frame)
        self.draw_countdown(frame)
        self.draw_feedback(frame)


#information for info button
class InfoButton:
    HINT = "Raise 2 fingers here to open instructions"

    def __init__(self, right_x, icons):
        self.icons = icons
        self.box = (right_x - BUTTON_WIDTH, TOOLBAR_TOP, right_x, TOOLBAR_TOP + TOOLBAR_HEIGHT)
        self.hovered = False

    def contains(self, point):
        return point_in_box(point, self.box)

    def update(self, gesture, point):
        self.hovered = gesture in ("draw", "cursor") and self.contains(point)
        return gesture == "cursor" and self.hovered

    def draw(self, frame):
        x1, y1, x2, y2 = self.box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = (y2 - y1) // 2
        if self.icons.has("info"):
            self.icons.draw(frame, "info", self.box)
        else:
            cv2.circle(frame, center, radius, (0, 0, 0), cv2.FILLED)
            TEXT.draw(frame, "i", center, "display", 48, centered=True)
        cv2.circle(frame, center, radius, (255, 255, 255), 2)
        if self.hovered:
            draw_selection(frame, self.box)
            TEXT.draw(frame, self.HINT, (center[0], y2 + 22), "semibold", 14, stroke=2, centered=True)


#keeps track of user behavior
class StatusDisplay:
    def draw(self, frame, palette, eraser, gesture, stylus):
        mode = "ERASER" if eraser.active else f"DRAW ({palette.selected_name} {stylus.size} {stylus.selected.name})"
        color = (255, 255, 255) if eraser.active else palette.selected_color
        TEXT.draw(frame, f"Mode: {mode}   Gesture: {gesture}", (20, frame.shape[0] - 38), "bold", 20, color, stroke=2)


#==============================================================================
# 3. MODES
# hand tracking + gesture detection, then cursor mode, fist-clear mode, welcome/calibration
#==============================================================================


#hand tracking and gesture detection classes
class HandTracker:
    WRIST = 0
    INDEX_PIP = 6
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_PIP = 14
    RING_TIP = 16
    PINKY_PIP = 18
    PINKY_TIP = 20

    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(find_or_download_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.start_time = time.monotonic()

    def find_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self.start_time) * 1000)
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None
        height, width = frame.shape[:2]
        return [(int(lm.x * width), int(lm.y * height)) for lm in result.hand_landmarks[0]]

    def draw_landmarks(self, frame, points):
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (255, 255, 255), cv2.FILLED)

    def close(self):
        self.landmarker.close()


#interprets hand gestures
class GestureDetector:
    def fingers_up(self, points):
        pairs = [
            (HandTracker.INDEX_TIP, HandTracker.INDEX_PIP),
            (HandTracker.MIDDLE_TIP, HandTracker.MIDDLE_PIP),
            (HandTracker.RING_TIP, HandTracker.RING_PIP),
            (HandTracker.PINKY_TIP, HandTracker.PINKY_PIP),
        ]
        wrist = points[HandTracker.WRIST]
        return [math.dist(wrist, points[tip]) > math.dist(wrist, points[pip]) for tip, pip in pairs]

    def classify(self, points):
        if points is None:
            return "none"
        fingers = self.fingers_up(points)
        if fingers == [False, False, False, False]:
            return "fist"
        if fingers == [True, True, False, False]:
            return "cursor"
        if fingers == [True, False, False, False]:
            return "draw"
        return "idle"


#tracking box for hand position and movement
class TrackingBox:
    BASE_PADDING = 60
    MAX_PADDING = 120
    SPEED_GAIN = 1.5
    SHRINK_RATE = 0.15
    LOST_GRACE_SECONDS = 0.3
    STABLE_FRAMES = 3
    EDGE_STABLE_FRAMES = 8

    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.reset()

    def reset(self):
        self.box = None
        self.last_points = None
        self.last_seen = None
        self.last_center = None
        self.gesture = "none"
        self.candidate = None
        self.candidate_frames = 0
        self.at_edge = False
        self.coasting = False

    def update(self, points, raw_gesture):
        now = time.monotonic()
        if points is None:
            # a brief detection dropout keeps the last hand instead of dropping the stroke
            if self.last_points is not None and now - self.last_seen <= self.LOST_GRACE_SECONDS:
                self.coasting = True
                return self.last_points, self.gesture
            self.reset()
            return None, "none"
        self.coasting = False
        self.last_seen = now
        self.last_points = points
        self.update_box(points)
        self.update_gesture(raw_gesture)
        return points, self.gesture

    def update_box(self, points):
        xs, ys = zip(*points)
        center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        speed = math.dist(center, self.last_center) if self.last_center else 0
        self.last_center = center
        padding = min(self.BASE_PADDING + speed * self.SPEED_GAIN, self.MAX_PADDING)
        target = (min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding)
        if self.box is None:
            self.box = target
        else:
            # grow instantly so every landmark stays inside, shrink gradually so the box doesn't jitter
            bx1, by1, bx2, by2 = self.box
            tx1, ty1, tx2, ty2 = target
            rate = self.SHRINK_RATE
            self.box = (
                min(tx1, bx1 + (tx1 - bx1) * rate),
                min(ty1, by1 + (ty1 - by1) * rate),
                max(tx2, bx2 + (tx2 - bx2) * rate),
                max(ty2, by2 + (ty2 - by2) * rate),
            )
        margin = self.BASE_PADDING
        self.at_edge = (
            min(xs) < margin
            or min(ys) < margin
            or max(xs) > self.frame_width - margin
            or max(ys) > self.frame_height - margin
        )

    def update_gesture(self, raw_gesture):
        if raw_gesture == self.gesture:
            self.candidate, self.candidate_frames = None, 0
            return
        if raw_gesture == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate, self.candidate_frames = raw_gesture, 1
        # near the camera edge fingers can be cut off, so a gesture change needs more confirming frames
        needed = self.EDGE_STABLE_FRAMES if self.at_edge else self.STABLE_FRAMES
        if self.candidate_frames >= needed:
            self.gesture = raw_gesture
            self.candidate, self.candidate_frames = None, 0


#mode to hover
class CursorMode:
    DOT_RADIUS = 7

    def draw(self, frame, point):
        cv2.circle(frame, point, self.DOT_RADIUS + 2, (0, 0, 0), cv2.FILLED)
        cv2.circle(frame, point, self.DOT_RADIUS, (255, 255, 255), cv2.FILLED)


#clears the whole screen
class ScreenClearer:
    COUNTDOWN_SECONDS = 3
    CLEARED_HOLD_SECONDS = 1.5
    GRACE_SECONDS = 0.2
    CANCEL_MESSAGE_SECONDS = 1.0

    IDLE = "idle"
    COUNTDOWN = "countdown"
    CLEARED = "cleared"

    def __init__(self):
        self.state = self.IDLE
        self.stage_start = None
        self.last_fist_seen = None
        self.cancel_message_until = 0

    @property
    def counting_down(self):
        return self.state == self.COUNTDOWN

    def seconds_left(self):
        return self.COUNTDOWN_SECONDS - (time.monotonic() - self.stage_start)

    def enter(self, state):
        self.state = state
        self.stage_start = time.monotonic()

    def fist_released(self, is_fist, now):
        if is_fist:
            self.last_fist_seen = now
            return False
        return self.last_fist_seen is None or now - self.last_fist_seen > self.GRACE_SECONDS

    def update(self, is_fist, canvas):
        now = time.monotonic()
        released = self.fist_released(is_fist, now)

        if self.state == self.IDLE:
            if is_fist:
                self.enter(self.COUNTDOWN)
            return False

        if released:
            if self.state == self.COUNTDOWN:
                self.cancel_message_until = now + self.CANCEL_MESSAGE_SECONDS
            self.state = self.IDLE
            self.last_fist_seen = None
            return False

        if self.state == self.COUNTDOWN and self.seconds_left() <= 0:
            canvas.clear()
            self.enter(self.CLEARED)
            return True

        if self.state == self.CLEARED and now - self.stage_start >= self.CLEARED_HOLD_SECONDS:
            self.enter(self.COUNTDOWN)
        return False

    def draw_countdown(self, frame):
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        remaining = self.seconds_left()
        number = str(max(math.ceil(remaining), 1))
        progress = 1 - max(remaining, 0) / self.COUNTDOWN_SECONDS
        cv2.circle(frame, center, 110, (0, 0, 0), cv2.FILLED)
        cv2.ellipse(frame, center, (100, 100), -90, 0, int(360 * progress), (0, 0, 255), 10)
        draw_centered(frame, number, center[1], "display", 140)
        draw_centered(frame, "Hold fist to clear - open hand to cancel", center[1] + 150, "bold", 24, stroke=3)

    def draw_cleared(self, frame):
        center_y = frame.shape[0] // 2
        draw_centered(frame, "Screen Cleared!", center_y, "display", 76, stroke=4)
        draw_centered(frame, "Keep holding to clear again - open hand to draw", center_y + 70, "semibold", 22, stroke=3)

    def draw(self, frame):
        if self.state == self.COUNTDOWN:
            self.draw_countdown(frame)
        elif self.state == self.CLEARED:
            self.draw_cleared(frame)
        elif time.monotonic() < self.cancel_message_until:
            draw_centered(frame, "Clear cancelled", frame.shape[0] // 2, "display", 54, stroke=4)


#welcome screen
class WelcomeScreen:
    CALIBRATION_SECONDS = 2.0
    GRACE_SECONDS = 0.2
    SUCCESS_SECONDS = 1.0
    PANEL_WIDTH = 280
    PANEL_HEIGHT = 300
    PANEL_GAP = 20
    PANEL_TOP = 185
    ACCENT = (0, 220, 255)
    SUCCESS = (0, 220, 0)

    def __init__(self, icons):
        self.icons = icons
        self.panels = [
            ("1. Use your index finger to draw.", "draw", [True, False, False, False]),
            ("2. Raise index + middle fingers to hover/select colors.", "cursor", [True, True, False, False]),
            ("3. Make a closed fist to clear the canvas.", "fist", [False, False, False, False]),
            ("4. Hover over the camera icon to save a screenshot.", "camera", None),
        ]
        self.hold_start = None
        self.last_seen = None
        self.success_time = None
        self.returning = False

    def restart(self):
        self.hold_start = None
        self.last_seen = None
        self.success_time = None
        self.returning = True

    @property
    def calibrated(self):
        return self.success_time is not None

    @property
    def complete(self):
        return self.calibrated and time.monotonic() - self.success_time >= self.SUCCESS_SECONDS

    def progress(self):
        if self.calibrated:
            return 1.0
        if self.hold_start is None:
            return 0.0
        return min((time.monotonic() - self.hold_start) / self.CALIBRATION_SECONDS, 1.0)

    def update(self, gesture):
        if self.calibrated:
            return
        now = time.monotonic()
        if gesture == "draw":
            self.last_seen = now
            if self.hold_start is None:
                self.hold_start = now
            if now - self.hold_start >= self.CALIBRATION_SECONDS:
                self.success_time = now
        elif self.last_seen is None or now - self.last_seen > self.GRACE_SECONDS:
            self.hold_start = None
            self.last_seen = None

    def draw_background(self, frame):
        dark = np.zeros_like(frame)
        frame[:] = cv2.addWeighted(frame, 0.25, dark, 0.75, 0)

    def draw_header(self, frame):
        draw_centered(frame, "Welcome to Air Canvas", 68, "display", 66)
        draw_centered(frame, "Drawing redefined", 140, "italic", 28, self.ACCENT)

    def draw_panels(self, frame):
        total = len(self.panels) * self.PANEL_WIDTH + (len(self.panels) - 1) * self.PANEL_GAP
        start_x = (frame.shape[1] - total) // 2
        for i, (text, icon, raised) in enumerate(self.panels):
            x1 = start_x + i * (self.PANEL_WIDTH + self.PANEL_GAP)
            x2 = x1 + self.PANEL_WIDTH
            y1, y2 = self.PANEL_TOP, self.PANEL_TOP + self.PANEL_HEIGHT
            region = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = cv2.addWeighted(region, 0.4, np.full_like(region, 50), 0.6, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 2)
            center_x = (x1 + x2) // 2
            if self.icons.has(icon):
                self.icons.draw(frame, icon, (x1 + 30, y1 + 20, x2 - 30, y1 + 185))
            elif icon == "camera":
                draw_camera_icon(frame, (center_x - 75, y1 + 55, center_x + 75, y1 + 155), (0, 200, 255))
            else:
                draw_hand_icon(frame, (center_x, y1 + 120), raised)
            for line_index, line in enumerate(wrap_text(text, self.PANEL_WIDTH - 30, "semibold", 19)):
                draw_centered(frame, line, y1 + 215 + line_index * 28, "semibold", 19, center_x=center_x)

    def draw_calibration(self, frame, gesture):
        if self.calibrated:
            message = "Calibration complete! Returning to your canvas..." if self.returning else "Calibration complete! Starting Air Canvas..."
            draw_centered(frame, message, 530, "bold", 26, self.SUCCESS)
        else:
            blink_on = int(time.monotonic() * 2) % 2 == 0
            color = self.ACCENT if blink_on else (255, 255, 255)
            action = "return to your canvas" if self.returning else "start"
            draw_centered(frame, f"(Hold up just your index finger to calibrate and {action})", 530, "bold", 24, color)

        bar_width = 500
        x1 = (frame.shape[1] - bar_width) // 2
        y1, y2 = 560, 580
        cv2.rectangle(frame, (x1, y1), (x1 + bar_width, y2), (80, 80, 80), cv2.FILLED)
        fill_color = self.SUCCESS if self.calibrated else self.ACCENT
        cv2.rectangle(frame, (x1, y1), (x1 + int(bar_width * self.progress()), y2), fill_color, cv2.FILLED)
        cv2.rectangle(frame, (x1, y1), (x1 + bar_width, y2), (255, 255, 255), 2)

        if gesture == "none":
            status = "No hand detected - show your hand to the camera"
        elif gesture == "draw":
            status = "Index finger detected - keep holding..."
        else:
            status = "Hand detected - raise only your index finger"
        if not self.calibrated:
            draw_centered(frame, status, 615, "regular", 18, (200, 200, 200))

    def draw_footer(self, frame):
        draw_centered(frame, "(press q to quit at any time)", 690, "regular", 16, (180, 180, 180))

    def render(self, frame, points, gesture, tracker):
        display = frame.copy()
        self.draw_background(display)
        if points:
            tracker.draw_landmarks(display, points)
        self.draw_header(display)
        self.draw_panels(display)
        self.draw_calibration(display, gesture)
        self.draw_footer(display)
        return display


class StylusProfile:
    #sizes optionally fixes the Thin/Regular/Thick widths for tips too fine for the usual multipliers
    def __init__(self, name, thickness, opacity, softness=0.0, wax=False, sizes=None):
        self.name = name
        self.thickness = thickness
        self.opacity = opacity
        self.softness = softness
        self.wax = wax
        self.sizes = sizes

    @property
    def blur(self):
        return self.thickness * self.softness

    @property
    def reach(self):
        return self.thickness // 2 + math.ceil(self.blur * 3) + 2

    #same stylus with a different tip width; softness is relative, so the edge scales with it
    def resized(self, thickness):
        return StylusProfile(self.name, thickness, self.opacity, self.softness, self.wax)

    def finish(self, mask, texture):
        if self.blur > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), self.blur)
        strength = mask.astype(np.float32) / 255
        if self.wax:
            strength = np.clip(strength * 1.5 - (1 - texture) * 1.2, 0, 1)
        return (strength * self.opacity).astype(np.uint8)


STYLUS_PROFILES = [
    StylusProfile("Pen", thickness=8, opacity=255),
    # halving and doubling 2px gives 1/2/4px, too close to tell apart, so the pencil gets its own steps
    StylusProfile("Pencil", thickness=4, opacity=225, sizes=(2, 4, 7)),
    StylusProfile("Highlighter", thickness=30, opacity=90),
    StylusProfile("Crayon", thickness=14, opacity=255, softness=0.12, wax=True),
    StylusProfile("Paint Brush", thickness=26, opacity=240, softness=0.25),
]


def make_wax_texture(width, height, seed=7):
    rng = np.random.default_rng(seed)
    fine = rng.random((height, width), dtype=np.float32)
    coarse = rng.random((height // 5 + 1, width // 5 + 1), dtype=np.float32)
    coarse = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(0.55 * fine + 0.45 * coarse, 0, 1)


def stamp_segment(layer, texture, start, end, profile):
    height, width = layer.shape[:2]
    reach = profile.reach
    x1 = max(min(start[0], end[0]) - reach, 0)
    y1 = max(min(start[1], end[1]) - reach, 0)
    x2 = min(max(start[0], end[0]) + reach + 1, width)
    y2 = min(max(start[1], end[1]) + reach + 1, height)
    if x1 >= x2 or y1 >= y2:
        return None
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.line(mask, (start[0] - x1, start[1] - y1), (end[0] - x1, end[1] - y1), 255, profile.thickness, cv2.LINE_AA)
    mask = profile.finish(mask, texture[y1:y2, x1:x2])
    region = layer[y1:y2, x1:x2]
    np.maximum(region, mask, out=region)
    return x1, y1, x2, y2


def union_bounds(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


class StylusCanvas(DrawingCanvas):
    def __init__(self, width, height, texture):
        super().__init__(width, height)
        self.texture = texture
        self.canvas = np.zeros((height, width, 4), dtype=np.uint8)
        self.stroke = np.zeros((height, width), dtype=np.uint8)
        self.stroke_color = None
        self.stroke_profile = None
        self.stroke_bounds = None
        self.painted_bounds = None

    def layers(self):
        return self.canvas, self.stroke

    def draw_line(self, point, color, profile):
        if (color, profile) != (self.stroke_color, self.stroke_profile):
            self.lift_pen()
            self.stroke_color, self.stroke_profile = color, profile
        if self.previous_point is None:
            self.previous_point = point
        bounds = stamp_segment(self.stroke, self.texture, self.previous_point, point, profile)
        self.stroke_bounds = union_bounds(self.stroke_bounds, bounds)
        self.previous_point = point

    def commit(self):
        if self.stroke_bounds is None:
            return
        x1, y1, x2, y2 = self.stroke_bounds
        region = self.canvas[y1:y2, x1:x2]
        alpha = self.stroke[y1:y2, x1:x2, None] / 255.0
        base_alpha = region[:, :, 3:] / 255.0
        out_alpha = alpha + base_alpha * (1 - alpha)
        mixed = np.array(self.stroke_color, dtype=np.float64) * alpha + region[:, :, :3] * base_alpha * (1 - alpha)
        region[:, :, :3] = np.divide(mixed, out_alpha, out=np.zeros_like(mixed), where=out_alpha > 0)
        region[:, :, 3:] = out_alpha * 255
        self.stroke[y1:y2, x1:x2] = 0
        self.painted_bounds = union_bounds(self.painted_bounds, self.stroke_bounds)
        self.stroke_bounds = None

    def lift_pen(self):
        self.commit()
        super().lift_pen()

    def clear(self):
        self.stroke[:] = 0
        self.stroke_bounds = None
        self.painted_bounds = None
        super().clear()

    def overlay(self, frame):
        result = frame.copy()
        if self.painted_bounds is not None:
            x1, y1, x2, y2 = self.painted_bounds
            blend(result, self.canvas[y1:y2, x1:x2], x1, y1)
        if self.stroke_bounds is not None:
            x1, y1, x2, y2 = self.stroke_bounds
            patch = np.empty((y2 - y1, x2 - x1, 4), dtype=np.uint8)
            patch[:, :, :3] = self.stroke_color
            patch[:, :, 3] = self.stroke[y1:y2, x1:x2]
            blend(result, patch, x1, y1)
        return result


class StylusMenu:
    ICON_SIZE = TOOLBAR_HEIGHT
    OPTION_WIDTH = 96
    OPTION_HEIGHT = 74
    OPTION_GAP = 10
    DROP_GAP = 8
    MARGIN = 20
    PANEL_PADDING = 6
    PANEL_SHADE = 30
    PANEL_ALPHA = 0.75
    CLOSE_DELAY = 0.8
    PAPER = (235, 240, 242)
    INK = (40, 40, 40)
    OPEN_COLOR = (0, 220, 255)
    # each size multiplies the stylus's own width, so a thick pencil is still thinner than a thick brush
    SIZES = [("Thin", 0.5), ("Regular", 1.0), ("Thick", 2.0)]
    SIZE_HEIGHT = 46
    SAMPLE_LENGTH = 38
    SAMPLE_MAX_WIDTH = 28

    def __init__(self, left_x, frame_width, profiles, texture, icons):
        self.profiles = profiles
        self.texture = texture
        self.icons = icons
        self.selected = profiles[0]
        self.size = "Regular"
        self.sized_profiles = {}
        self.is_open = False
        self.away_since = None
        self.previews = {}
        self.box = (left_x, TOOLBAR_TOP, left_x + self.ICON_SIZE, TOOLBAR_TOP + TOOLBAR_HEIGHT)
        row_width = len(profiles) * (self.OPTION_WIDTH + self.OPTION_GAP) - self.OPTION_GAP
        center_x = (self.box[0] + self.box[2]) // 2
        left = min(max(center_x - row_width // 2, self.MARGIN), frame_width - self.MARGIN - row_width)
        top = self.box[3] + self.DROP_GAP
        bottom = top + self.OPTION_HEIGHT
        self.options = []
        for i, profile in enumerate(profiles):
            x1 = left + i * (self.OPTION_WIDTH + self.OPTION_GAP)
            self.options.append((profile, (x1, top, x1 + self.OPTION_WIDTH, bottom)))
        # a second row under the styluses for the stroke size
        size_top = bottom + self.OPTION_GAP
        size_width = (row_width - (len(self.SIZES) - 1) * self.OPTION_GAP) // len(self.SIZES)
        self.size_options = []
        for i, (name, _) in enumerate(self.SIZES):
            x1 = left + i * (size_width + self.OPTION_GAP)
            self.size_options.append((name, (x1, size_top, x1 + size_width, size_top + self.SIZE_HEIGHT)))
        pad = self.PANEL_PADDING
        self.panel = (left - pad, top - pad, left + row_width + pad, size_top + self.SIZE_HEIGHT + pad)

    def contains(self, point):
        return point_in_box(point, self.box) or (self.is_open and point_in_box(point, self.panel))

    def target_at(self, point):
        if self.is_open:
            for profile, box in self.options:
                if point_in_box(point, box):
                    return ("brush", profile.name), box
            for name, box in self.size_options:
                if point_in_box(point, box):
                    return ("size", name), box
            return None
        if point_in_box(point, self.box):
            return ("stylus", None), self.box
        return None

    def open(self):
        self.is_open = True
        self.away_since = None

    def close(self):
        self.is_open = False
        self.away_since = None

    #picking a stylus or a size keeps the menu open so the other can be picked too;
    #it closes on its own once the hand moves away
    def select(self, name):
        for profile in self.profiles:
            if profile.name == name:
                self.selected = profile

    def select_size(self, name):
        self.size = name

    @property
    def size_factor(self):
        return dict(self.SIZES)[self.size]

    #the selected stylus at the selected size; cached so the canvas sees the same profile every frame
    @property
    def current(self):
        key = (self.selected.name, self.size)
        if key not in self.sized_profiles:
            self.sized_profiles[key] = self.selected.resized(self.width_for(self.selected, self.size))
        return self.sized_profiles[key]

    #a stylus's width at a size: its own fixed steps if it has them, otherwise its width times the multiplier
    def width_for(self, profile, size):
        names = [name for name, _ in self.SIZES]
        if profile.sizes:
            return profile.sizes[names.index(size)]
        return max(1, round(profile.thickness * dict(self.SIZES)[size]))

    def update(self, gesture, point):
        if not self.is_open:
            return
        if gesture == "cursor" and self.contains(point):
            self.away_since = None
            return
        now = time.monotonic()
        if self.away_since is None:
            self.away_since = now
        elif now - self.away_since >= self.CLOSE_DELAY:
            self.close()

    def preview(self, profile, color, width, height):
        key = (profile.name, color, width, height)
        if key not in self.previews:
            layer = np.zeros((height, width), dtype=np.uint8)
            inset = min(profile.thickness // 2 + 4, width // 4)
            amplitude = max((height - profile.thickness) // 2 - 4, 2)
            span = width - 2 * inset
            points = [
                (inset + span * i // 12, int(height / 2 + amplitude * math.sin(i / 12 * 2 * math.pi)))
                for i in range(13)
            ]
            for start, end in zip(points, points[1:]):
                stamp_segment(layer, self.texture, start, end, profile)
            tile = np.empty((height, width, 4), dtype=np.uint8)
            tile[:, :, :3] = color
            tile[:, :, 3] = layer
            self.previews[key] = tile
        return self.previews[key]

    def draw_tile(self, frame, profile, color, box, label_size):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.PAPER, cv2.FILLED)
        px1, py1, px2, py2 = x1 + 6, y1 + 6, x2 - 6, y2 - 22
        if self.icons.has(profile.name):
            self.icons.draw(frame, (profile.name, color), (px1, py1, px2, py2))
        else:
            blend(frame, self.preview(profile, color, px2 - px1, py2 - py1), px1, py1)
        TEXT.draw(frame, profile.name, ((x1 + x2) // 2, y2 - 12), "display", label_size, self.INK, centered=True)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.INK, 2)

    #a short sample stroke for this size: true width when it fits, otherwise
    #scaled so Thick fills the sample space and the three sizes still look different
    def draw_size(self, frame, name, color, box, thickness):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.PAPER, cv2.FILLED)
        factor = dict(self.SIZES)[name]
        largest = max(f for _, f in self.SIZES)
        width = max(round(min(thickness, self.SAMPLE_MAX_WIDTH * factor / largest)), 1)
        center_y = (y1 + y2) // 2
        # the line's round caps reach width/2 past each end, so the ends move in to stay in the sample space
        cap = width // 2
        start, end = x1 + 10 + cap, x1 + 10 + self.SAMPLE_LENGTH - cap
        cv2.line(frame, (start, center_y), (max(end, start), center_y), color, width, cv2.LINE_AA)
        label_x = (x1 + 10 + self.SAMPLE_LENGTH + 4 + x2) // 2
        TEXT.draw(frame, f"{name} {thickness}px", (label_x, center_y), "display", 15, self.INK, centered=True)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.INK, 2)

    def draw_panel(self, frame):
        x1, y1, x2, y2 = self.panel
        region = frame[y1:y2, x1:x2]
        shade = np.full_like(region, self.PANEL_SHADE)
        region[:] = cv2.addWeighted(region, 1 - self.PANEL_ALPHA, shade, self.PANEL_ALPHA, 0)

    #base_width replaces the stylus's own width in the size labels, e.g. the outline's fill width while tracing
    def draw(self, frame, color, base_width=None):
        if self.is_open:
            self.draw_panel(frame)
            for profile, box in self.options:
                self.draw_tile(frame, profile, color, box, 13)
                if profile is self.selected:
                    draw_selection(frame, box)
            for name, box in self.size_options:
                if base_width:
                    thickness = max(1, round(base_width * dict(self.SIZES)[name]))
                else:
                    thickness = self.width_for(self.selected, name)
                self.draw_size(frame, name, color, box, thickness)
                if name == self.size:
                    draw_selection(frame, box)
        self.draw_tile(frame, self.selected, color, self.box, 11)
        if self.is_open:
            draw_selection(frame, self.box, self.OPEN_COLOR)

    #names the stylus in use under its button, with the width it actually draws at (set by the outline while tracing)
    def draw_caption(self, frame, profile):
        if self.is_open:
            return
        x1, _, x2, y2 = self.box
        TEXT.draw(frame, f"{profile.name} {profile.thickness}px", ((x1 + x2) // 2, y2 + 16), "semibold", 14, stroke=2, centered=True)


#==============================================================================
# 4. TRACE & PLAY
# type a name or pick an emoji, and a gray outline appears behind the drawing to trace over
#==============================================================================


#picker tabs as hex codepoint ranges; the first tab that matches wins and anything unmatched is an object
EMOJI_CATEGORIES = [
    ("Faces", "261D 2639-263A 270A-270D 1F3C3 1F440-1F450 1F464-1F487 1F48B 1F48F 1F491 1F4A9-1F4AA "
              "1F574-1F575 1F57A 1F590-1F596 1F5E3 1F600-1F64F 1F6B6 1F90C 1F90F 1F910-1F939 1F970-1F97A "
              "1F9B5-1F9B9 1F9BB 1F9BE-1F9BF 1F9CC-1F9E0 1FAC0-1FAC6 1FAE0-1FAF8"),
    ("Nature", "2600-2601 2603-2604 2614 2618 26A1 26C4-26C8 2744 2B50 1F300 1F308 1F30A 1F30C 1F311-1F321 "
               "1F324-1F32C 1F330-1F344 1F3F5 1F400-1F43F 1F490 1F4A7 1F4AE 1F525 1F54A 1F577-1F578 1F940 "
               "1F980-1F9AE 1FAA8 1FAB0-1FABF 1FACE-1FACF"),
    ("Food", "2615 1F32D-1F32F 1F345-1F37F 1F942-1F944 1F950-1F96F 1F9C0-1F9CB 1FAD0-1FADC"),
    ("Travel", "2668 2693 26E9-26EA 26F0-26F2 26F4-26F5 26FA 26FD 2708 1F301 1F303-1F307 1F309 1F30B "
               "1F30D-1F310 1F3A0-1F3A2 1F3D4-1F3F0 1F488 1F492 1F54B-1F54D 1F5FA-1F5FF 1F680-1F6A8 1F6B2 "
               "1F6D5-1F6D6 1F6DD-1F6DF 1F6E3-1F6E5 1F6E9-1F6EC 1F6F0-1F6FC 1F9F3 1FA90"),
    ("Activities", "265F-2660 2663 2665-2666 26BD-26BE 26F3 26F7-26F9 1F004 1F0CF 1F380-1F38B 1F38D-1F393 "
                   "1F396-1F397 1F39F 1F3A3 1F3A8 1F3AA-1F3B4 1F3BD-1F3D3 1F3F8 1F52E 1F579 1F93A-1F93F "
                   "1F945-1F94F 1F9E7-1F9E9 1F9F5-1F9F6 1F9F8 1F9FF 1FA80-1FA87 1FAA9"),
    ("Objects", "231A-231B 2328 23F0-23F3 2602 260E 2692 2694-2697 2699 26B0-26B1 26CF 26D1 26D3 2702 2709 "
                "270F 2712"),
    ("Symbols", "2000-2BFF 3000-33FF 1F170-1F251 1F38C 1F3F3-1F3F4 1F493-1F4A0 1F4A2 1F4A4-1F4A6 1F4A8 "
                "1F4AB-1F4AF 1F4B1-1F4B2 1F4F3-1F4F6 1F500-1F506 1F515 1F518-1F524 1F52F-1F53D 1F549 1F54E "
                "1F5A4 1F5E8 1F5EF 1F6A9 1F6AB-1F6B1 1F6B3 1F6B7-1F6BC 1F6BE 1F6C2-1F6C5 1F6D0-1F6D1 "
                "1F7E0-1F7F0 1F90D-1F90E 1F9E1 1FA75-1FA77"),
]
EMOJI_SCAN = "2000-2BFF 3000-33FF 1F000-1FAFF"
# flag letters, skin tones and hair styles only mean something when combined with another emoji
EMOJI_SKIP = "1F1E6-1F1FF 1F3FB-1F3FF 1F9B0-1F9B3"


#"1F600-1F64F 263A" -> [(0x1F600, 0x1F64F), (0x263A, 0x263A)]
def parse_ranges(spec):
    ranges = []
    for part in spec.split():
        low, _, high = part.partition("-")
        ranges.append((int(low, 16), int(high or low, 16)))
    return ranges


def in_ranges(code, ranges):
    return any(low <= code <= high for low, high in ranges)


#color emoji font for the picker tiles and for emoji outlines
class EmojiSet:
    # color emoji fonts only come in fixed bitmap sizes
    FONTS = [
        ("/System/Library/Fonts/Apple Color Emoji.ttc", 160),
        ("C:/Windows/Fonts/seguiemj.ttf", 160),
        ("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109),
    ]
    PROBE_SIZE = 20

    def __init__(self):
        self.font = None
        self.probe = None
        self.images = {}
        self.tiles = {}
        self.categories = None
        self.names = {}
        for path, size in self.FONTS:
            try:
                self.font = ImageFont.truetype(path, size)
            except OSError:
                continue
            # checking glyphs at a small size is what keeps the full scan fast; single-size fonts use their one size
            try:
                self.probe = ImageFont.truetype(path, self.PROBE_SIZE)
            except OSError:
                self.probe = self.font
            break
        if self.font is None:
            print("No color emoji font found - Trace & Play offers names only")

    @property
    def available(self):
        return self.font is not None

    #every single-character emoji the font can draw, sorted into the picker tabs; built on first use
    @property
    def catalog(self):
        if self.categories is None:
            tabs = [(name, parse_ranges(spec)) for name, spec in EMOJI_CATEGORIES]
            skip = parse_ranges(EMOJI_SKIP)
            self.categories = {name: [] for name, _ in tabs}
            for low, high in parse_ranges(EMOJI_SCAN):
                for code in range(low, high + 1):
                    emoji = chr(code)
                    if in_ranges(code, skip) or not self.has_glyph(emoji):
                        continue
                    tab = next((name for name, ranges in tabs if in_ranges(code, ranges)), "Objects")
                    self.categories[tab].append(emoji)
                    self.names[emoji] = unicodedata.name(emoji, "").lower().replace("-", " ").split()
        return self.categories

    def has_glyph(self, emoji):
        _, top, _, bottom = self.probe.getbbox(emoji)
        return bottom > top

    #emoji whose Unicode name has a word starting with each typed word, so "red hea" finds the red heart
    def search(self, text):
        words = text.lower().split()
        return [
            emoji for emojis in self.catalog.values() for emoji in emojis
            if all(any(part.startswith(word) for part in self.names[emoji]) for word in words)
        ]

    #the emoji as a BGRA image cropped to its visible pixels
    def render(self, emoji):
        if emoji not in self.images:
            x1, y1, x2, y2 = self.font.getbbox(emoji)
            image = Image.new("RGBA", (x2 - x1, y2 - y1), (0, 0, 0, 0))
            ImageDraw.Draw(image).text((-x1, -y1), emoji, font=self.font, embedded_color=True)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
            ys, xs = np.nonzero(image[:, :, 3])
            self.images[emoji] = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return self.images[emoji]

    def tile(self, emoji, size):
        key = (emoji, size)
        if key not in self.tiles:
            image = self.render(emoji)
            scale = size / max(image.shape[:2])
            self.tiles[key] = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return self.tiles[key]


#scales an image to fit inside a width x height box
def fit_image(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)


#the outer edge of every letter, plus the holes in letters like "a" and "o"
def text_lines(text, width, height):
    font = TEXT.font("bold", 220)
    x1, y1, x2, y2 = font.getbbox(text)
    image = Image.new("L", (x2 - x1 + 20, y2 - y1 + 20), 0)
    ImageDraw.Draw(image).text((10 - x1, 10 - y1), text, font=font, fill=255)
    mask = fit_image(np.array(image), width, height)
    contours, _ = cv2.findContours((mask > 128).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return [cv2.approxPolyDP(c, 1.5, True) for c in contours], mask.shape


#the emoji's silhouette plus the edges between its main color regions
def emoji_lines(image, width, height):
    image = fit_image(image, width, height)
    solid = (image[:, :, 3] > 128).astype(np.uint8)
    # flattening the colors first drops the shading so only real feature lines remain
    flat = cv2.pyrMeanShiftFiltering(cv2.bitwise_and(image[:, :, :3], image[:, :, :3], mask=solid * 255), 16, 60)
    edges = cv2.Canny(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), 80, 180)
    edges[cv2.erode(solid, np.ones((9, 9), np.uint8)) == 0] = 0
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    outer, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    inner, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    inner = [c for c in inner if cv2.arcLength(c, True) > 70]
    return [cv2.approxPolyDP(c, 1.5, True) for c in [*outer, *inner]], solid.shape


#static gray outline on its own layer, under the drawing so the eraser and fist-clear leave it alone
class TraceOutline:
    LINE = (200, 200, 200)
    HALO = (70, 70, 70)
    LINE_THICKNESS = 3
    HALO_THICKNESS = 7
    # while tracing every stylus draws at 1/12 of the outline's size, so switching styluses keeps the width
    FILL_DIVISOR = 12
    MIN_FILL_THICKNESS = 8

    def __init__(self, width, height, box, emoji):
        self.layer = np.zeros((height, width, 4), dtype=np.uint8)
        self.box = box
        self.emoji = emoji
        self.bounds = None
        self.fill_thickness = None
        self.fill_profiles = {}

    @property
    def active(self):
        return self.bounds is not None

    def clear(self):
        self.layer[:] = 0
        self.bounds = None
        self.fill_thickness = None
        self.fill_profiles = {}

    #the stylus at the outline's fill width, keeping its own look (opacity, softness, wax);
    #cached so the canvas sees the same profile every frame and keeps the stroke going
    def fill_profile(self, profile, factor=1.0):
        key = (profile.name, factor)
        if key not in self.fill_profiles:
            self.fill_profiles[key] = profile.resized(max(1, round(self.fill_thickness * factor)))
        return self.fill_profiles[key]

    def set_text(self, text):
        x1, y1, x2, y2 = self.box
        self.place(*text_lines(text, x2 - x1, y2 - y1))

    def set_emoji(self, emoji):
        x1, y1, x2, y2 = self.box
        self.place(*emoji_lines(self.emoji.render(emoji), x2 - x1, y2 - y1))

    #centers the traced lines in the box; the dark halo keeps light gray visible on any background
    def place(self, lines, shape):
        self.clear()
        x1, y1, x2, y2 = self.box
        height, width = shape
        left, top = x1 + (x2 - x1 - width) // 2, y1 + (y2 - y1 - height) // 2
        lines = [line + (left, top) for line in lines]
        cv2.polylines(self.layer, lines, True, (*self.HALO, 255), self.HALO_THICKNESS, cv2.LINE_AA)
        cv2.polylines(self.layer, lines, True, (*self.LINE, 255), self.LINE_THICKNESS, cv2.LINE_AA)
        pad = self.HALO_THICKNESS
        self.bounds = (left - pad, top - pad, left + width + pad, top + height + pad)
        self.fill_thickness = max(round(min(width, height) / self.FILL_DIVISOR), self.MIN_FILL_THICKNESS)

    def draw(self, frame):
        if self.bounds is None:
            return
        x1, y1, x2, y2 = self.bounds
        blend(frame, self.layer[y1:y2, x1:x2], x1, y1)


#button under the screenshot camera that opens the Trace & Play overlay
class TraceButton:
    HEIGHT = 60
    GAP = 35

    def __init__(self, camera_box):
        x1, _, x2, y2 = camera_box
        self.box = (x1, y2 + self.GAP, x2, y2 + self.GAP + self.HEIGHT)

    def contains(self, point):
        return point_in_box(point, self.box)

    def draw(self, frame):
        x1, y1, x2, y2 = self.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (45, 45, 45), cv2.FILLED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        TEXT.draw(frame, "Trace & Play", ((x1 + x2) // 2, (y1 + y2) // 2), "display", 16, stroke=2, centered=True)


#buttons floating in the canvas's top-left while an outline is up: drop the outline and keep drawing,
#or start over with a new icon; they sit inside the edge buffer so reaching them never sets off a warning
class TraceControls:
    WIDTH = 170
    HEIGHT = 50
    GAP = 12
    INSET = 10
    ACTIONS = [("clear_outline", "Clear Outline"), ("change_icon", "Change Icon")]

    def __init__(self, safe_box):
        left, top = safe_box[0] + self.INSET, safe_box[1] + self.INSET
        self.buttons = []
        for i, (action, label) in enumerate(self.ACTIONS):
            y1 = top + i * (self.HEIGHT + self.GAP)
            self.buttons.append((action, label, (left, y1, left + self.WIDTH, y1 + self.HEIGHT)))
        self.right = left + self.WIDTH

    def target_at(self, point):
        for action, _, box in self.buttons:
            if point_in_box(point, box):
                return (action, None), box
        return None

    def contains(self, point):
        return self.target_at(point) is not None

    def draw(self, frame):
        for action, label, box in self.buttons:
            x1, y1, x2, y2 = box
            border = (0, 220, 255) if action == "change_icon" else (255, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (45, 45, 45), cv2.FILLED)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2)
            TEXT.draw(frame, label, ((x1 + x2) // 2, (y1 + y2) // 2), "display", 21, stroke=2, centered=True)


#input overlay: every emoji the font has, in tabs and pages, searchable by typing;
#hovering an emoji traces it, and Enter traces the typed word instead
class TraceMenu:
    MAX_LENGTH = 14
    FIELD_WIDTH = 640
    FIELD_TOP = 132
    FIELD_HEIGHT = 56
    TAB_WIDTH = 140
    TAB_HEIGHT = 44
    TAB_GAP = 10
    TAB_TOP = 210
    COLUMNS = 11
    ROWS = 3
    TILE = 76
    TILE_GAP = 12
    GRID_TOP = 280
    ARROW_WIDTH = 60
    ARROW_GAP = 20
    BUTTON_WIDTH = 190
    BUTTON_HEIGHT = 52
    BUTTON_GAP = 24
    BUTTON_TOP = 590
    ACCENT = (0, 220, 255)
    PANEL = (60, 60, 60)
    INK = (40, 40, 40)
    ENTER_KEYS = (10, 13)
    BACKSPACE_KEYS = (8, 127)
    ESCAPE_KEY = 27

    def __init__(self, frame_width, emoji):
        self.emoji = emoji
        self.selector = HoverSelector()
        self.is_open = False
        self.text = ""
        self.category = EMOJI_CATEGORIES[0][0]
        self.page = 0
        center_x = frame_width // 2
        self.field = (center_x - self.FIELD_WIDTH // 2, self.FIELD_TOP, center_x + self.FIELD_WIDTH // 2, self.FIELD_TOP + self.FIELD_HEIGHT)
        self.tabs = []
        self.slots = []
        self.arrows = []
        if emoji.available:
            row_width = len(EMOJI_CATEGORIES) * (self.TAB_WIDTH + self.TAB_GAP) - self.TAB_GAP
            for i, (name, _) in enumerate(EMOJI_CATEGORIES):
                x1 = center_x - row_width // 2 + i * (self.TAB_WIDTH + self.TAB_GAP)
                self.tabs.append((name, (x1, self.TAB_TOP, x1 + self.TAB_WIDTH, self.TAB_TOP + self.TAB_HEIGHT)))
            step = self.TILE + self.TILE_GAP
            grid_left = center_x - (self.COLUMNS * step - self.TILE_GAP) // 2
            for i in range(self.COLUMNS * self.ROWS):
                x1, y1 = grid_left + (i % self.COLUMNS) * step, self.GRID_TOP + (i // self.COLUMNS) * step
                self.slots.append((x1, y1, x1 + self.TILE, y1 + self.TILE))
            grid_bottom = self.GRID_TOP + self.ROWS * step - self.TILE_GAP
            left_x2 = grid_left - self.ARROW_GAP
            right_x1 = frame_width - left_x2
            self.arrows = [
                (-1, (left_x2 - self.ARROW_WIDTH, self.GRID_TOP, left_x2, grid_bottom)),
                (1, (right_x1, self.GRID_TOP, right_x1 + self.ARROW_WIDTH, grid_bottom)),
            ]
            self.status_y = grid_bottom + 24
        labels = [("cancel", "Cancel"), ("submit", "Trace word")]
        row_width = len(labels) * (self.BUTTON_WIDTH + self.BUTTON_GAP) - self.BUTTON_GAP
        self.buttons = []
        for i, (action, label) in enumerate(labels):
            x1 = center_x - row_width // 2 + i * (self.BUTTON_WIDTH + self.BUTTON_GAP)
            self.buttons.append((action, label, (x1, self.BUTTON_TOP, x1 + self.BUTTON_WIDTH, self.BUTTON_TOP + self.BUTTON_HEIGHT)))

    def open(self):
        self.is_open = True
        self.text = ""
        self.page = 0
        self.selector.reset()

    def close(self):
        self.is_open = False
        self.selector.reset()

    @property
    def searching(self):
        return bool(self.text.strip())

    #the emoji behind the grid: search results while typing, otherwise the chosen tab
    def results(self):
        if not self.emoji.available:
            return []
        if self.searching:
            return self.emoji.search(self.text)
        return self.emoji.catalog[self.category]

    def page_count(self, results):
        return max(1, math.ceil(len(results) / len(self.slots)))

    def visible(self, results):
        start = self.page * len(self.slots)
        return list(zip(results[start:start + len(self.slots)], self.slots))

    def enabled(self, action):
        if action == "submit":
            return self.searching
        return True

    def target_at(self, point):
        results = self.results()
        for symbol, box in self.visible(results):
            if point_in_box(point, box):
                return ("emoji", symbol), box
        for name, box in self.tabs:
            if point_in_box(point, box):
                return ("tab", name), box
        for step, box in self.arrows:
            if 0 <= self.page + step < self.page_count(results) and point_in_box(point, box):
                return ("page", step), box
        for action, _, box in self.buttons:
            if self.enabled(action) and point_in_box(point, box):
                return (action, self.text.strip()), box
        return None, None

    #hovering with two fingers switches tabs and pages, picks an emoji, or presses a button;
    #only picks and button presses are handed back to the app
    def update(self, gesture, point):
        target, box = self.target_at(point) if gesture == "cursor" else (None, None)
        confirmed = self.selector.update(target, box)
        if confirmed is None:
            return None
        kind, value = confirmed
        if kind == "tab":
            self.text = ""
            self.category = value
            self.page = 0
            return None
        if kind == "page":
            self.page += value
            return None
        return confirmed

    #typing searches the emoji; Enter traces the word and Escape closes the overlay
    def handle_key(self, key):
        if key in self.ENTER_KEYS:
            return ("submit", self.text.strip()) if self.searching else None
        if key == self.ESCAPE_KEY:
            return ("cancel", None)
        if key in self.BACKSPACE_KEYS:
            self.text = self.text[:-1]
            self.page = 0
        elif 32 <= key <= 126 and len(self.text) < self.MAX_LENGTH:
            self.text += chr(key)
            self.page = 0
        return None

    def draw_field(self, frame):
        x1, y1, x2, y2 = self.field
        cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), cv2.FILLED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.ACCENT, 3)
        center_y = (y1 + y2) // 2
        placeholder = "Search emoji or type a word..." if self.tabs else "Type a name..."
        if self.text:
            TEXT.draw(frame, self.text, ((x1 + x2) // 2, center_y), "bold", 30, self.INK, centered=True)
            caret_x = (x1 + x2) // 2 + TEXT.measure(self.text, "bold", 30)[0] // 2 + 6
        else:
            TEXT.draw(frame, placeholder, ((x1 + x2) // 2, center_y), "semibold", 24, (160, 160, 160), centered=True)
            caret_x = (x1 + x2) // 2 - TEXT.measure(placeholder, "semibold", 24)[0] // 2 - 8
        if int(time.monotonic() * 2) % 2 == 0:
            cv2.line(frame, (caret_x, center_y - 18), (caret_x, center_y + 18), self.INK, 2)

    def draw_tabs(self, frame):
        for name, box in self.tabs:
            x1, y1, x2, y2 = box
            chosen = name == self.category and not self.searching
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.ACCENT if chosen else self.PANEL, cv2.FILLED)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 2)
            color = self.INK if chosen else (255, 255, 255)
            TEXT.draw(frame, name, ((x1 + x2) // 2, (y1 + y2) // 2), "display", 20, color, centered=True)

    def draw_arrow(self, frame, step, box, enabled):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.PANEL, cv2.FILLED)
        color = (255, 255, 255) if enabled else (110, 110, 110)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        center_x, center_y, half = (x1 + x2) // 2, (y1 + y2) // 2, (x2 - x1) // 4
        tip = center_x + step * half
        points = np.array([(tip, center_y), (tip - step * 2 * half, center_y - 2 * half), (tip - step * 2 * half, center_y + 2 * half)])
        cv2.fillPoly(frame, [points], color, cv2.LINE_AA)

    def draw_grid(self, frame):
        results = self.results()
        pages = self.page_count(results)
        for symbol, box in self.visible(results):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.PANEL, cv2.FILLED)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 2)
            tile = self.emoji.tile(symbol, self.TILE - 18)
            blend(frame, tile, x1 + (x2 - x1 - tile.shape[1]) // 2, y1 + (y2 - y1 - tile.shape[0]) // 2)
        for step, box in self.arrows:
            self.draw_arrow(frame, step, box, 0 <= self.page + step < pages)
        if self.searching and not results:
            status = f'No emoji named "{self.text.strip()}" - press Enter to trace the word'
            draw_centered(frame, status, self.GRID_TOP + 120, "semibold", 22, (200, 200, 200))
            return
        found = f'{len(results)} found for "{self.text.strip()}"' if self.searching else self.category
        draw_centered(frame, f"{found}  -  page {self.page + 1} of {pages}", self.status_y, "semibold", 18, (200, 200, 200))

    def draw_buttons(self, frame):
        for action, label, box in self.buttons:
            x1, y1, x2, y2 = box
            enabled = self.enabled(action)
            fill = self.ACCENT if action == "submit" and enabled else self.PANEL
            cv2.rectangle(frame, (x1, y1), (x2, y2), fill, cv2.FILLED)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255) if enabled else (110, 110, 110), 2)
            color = self.INK if fill == self.ACCENT else (255, 255, 255) if enabled else (130, 130, 130)
            TEXT.draw(frame, label, ((x1 + x2) // 2, (y1 + y2) // 2), "display", 22, color, centered=True)

    def draw(self, frame, point):
        dark = np.zeros_like(frame)
        frame[:] = cv2.addWeighted(frame, 0.3, dark, 0.7, 0)
        draw_centered(frame, "Trace & Play", 58, "display", 50)
        hint = "Hover an emoji to trace it, or type a word and press Enter" if self.tabs else "Type a name and press Enter"
        draw_centered(frame, hint, 108, "semibold", 22, self.ACCENT)
        self.draw_field(frame)
        if self.tabs:
            self.draw_tabs(frame)
            self.draw_grid(frame)
        self.draw_buttons(frame)
        draw_centered(frame, "(Esc to close)", 680, "regular", 16, (180, 180, 180))
        self.selector.draw(frame, point)


#==============================================================================
# APP
# creates every object above and runs the main loop
#==============================================================================


#runs the AirCanvas application
class AirCanvasApp:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.tracker = HandTracker()
        self.gestures = GestureDetector()
        self.tracking_box = TrackingBox(FRAME_WIDTH, FRAME_HEIGHT)
        self.icons = IconSet()
        self.palette = ColorPalette(self.icons)
        self.wax = make_wax_texture(FRAME_WIDTH, FRAME_HEIGHT)
        self.stylus = StylusMenu(self.palette.right_edge + BUTTON_GAP, FRAME_WIDTH, STYLUS_PROFILES, self.wax, StylusIconSet())
        self.eraser = Eraser(self.stylus.box[2] + BUTTON_GAP, self.icons)
        self.zone = DrawingZone(FRAME_WIDTH, FRAME_HEIGHT)
        self.safe_zones = SafeZones(FRAME_WIDTH, FRAME_HEIGHT, self.zone)
        self.edge_warning = EdgeWarning(self.zone, self.safe_zones)
        self.selector = HoverSelector()
        self.canvas = StylusCanvas(FRAME_WIDTH, FRAME_HEIGHT, self.wax)
        self.clearer = ScreenClearer()
        self.camera = ScreenshotCamera(FRAME_WIDTH, DOWNLOADS_DIR, self.icons)
        self.info = InfoButton(self.camera.box[0] - BUTTON_GAP, self.icons)
        self.cursor = CursorMode()
        self.status = StatusDisplay()
        self.welcome = WelcomeScreen(self.icons)
        self.emoji = EmojiSet()
        x1, y1, x2, y2 = self.edge_warning.safe_box
        self.trace_controls = TraceControls(self.edge_warning.safe_box)
        # the outline stays centered but keeps clear of the floating buttons on the left
        side = self.trace_controls.right - x1 + 20
        self.outline = TraceOutline(FRAME_WIDTH, FRAME_HEIGHT, (x1 + side, y1 + 20, x2 - side, y2 - 20), self.emoji)
        self.trace_button = TraceButton(self.camera.box)
        self.trace_menu = TraceMenu(FRAME_WIDTH, self.emoji)

    def read_frame(self):
        success, frame = self.capture.read()
        if not success:
            return None
        frame = cv2.flip(frame, 1)
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        return frame

    def find_toolbar_target(self, gesture, index_tip):
        if gesture == "cursor":
            palette_hit = self.palette.target_at(index_tip)
            if palette_hit:
                return palette_hit
            stylus_hit = self.stylus.target_at(index_tip)
            if stylus_hit:
                return stylus_hit
            # the outline swaps the Trace & Play button for Clear Outline and Change Icon
            if self.outline.active:
                control_hit = self.trace_controls.target_at(index_tip)
                if control_hit:
                    return control_hit
            elif self.trace_button.contains(index_tip):
                return ("trace", None), self.trace_button.box
        if gesture in ("draw", "cursor") and self.eraser.contains(index_tip):
            return ("eraser", None), self.eraser.box
        return None, None

    def handle_toolbar(self, gesture, index_tip):
        pointing = gesture in ("draw", "cursor")
        self.camera.update_press(index_tip if pointing else None)
        self.palette.update(gesture, index_tip)
        self.stylus.update(gesture, index_tip)
        target, box = self.find_toolbar_target(gesture, index_tip)
        confirmed = self.selector.update(target, box)
        if confirmed is None:
            return
        kind, name = confirmed
        if kind == "color":
            # hovering a color reveals its shades; a new color family is also picked right away,
            # while the active family keeps its shade until one is chosen from the dropdown
            if name != self.palette.selected_family:
                self.palette.select(name)
            self.palette.open(name)
            self.stylus.close()
            self.eraser.active = False
        elif kind == "shade":
            self.palette.select_shade(name)
            self.eraser.active = False
        elif kind == "stylus":
            self.palette.close()
            self.stylus.open()
        elif kind == "brush":
            self.stylus.select(name)
            self.eraser.active = False
        elif kind == "size":
            self.stylus.select_size(name)
            self.eraser.active = False
        elif kind == "trace":
            self.open_trace()
        elif kind == "clear_outline":
            self.outline.clear()
        elif kind == "change_icon":
            self.change_icon()
        else:
            self.eraser.active = True

    def handle_painting(self, gesture, index_tip):
        on_button = (
            self.palette.contains(index_tip)
            or self.eraser.contains(index_tip)
            or self.camera.contains(index_tip)
            or self.info.contains(index_tip)
            or self.stylus.contains(index_tip)
            or (self.outline.active and self.trace_controls.contains(index_tip))
            or (not self.outline.active and self.trace_button.contains(index_tip))
        )
        self.edge_warning.update(gesture, index_tip, on_button)
        can_paint = (
            gesture == "draw"
            and self.zone.contains(index_tip)
            and not self.edge_warning.restricted
            and not self.safe_zones.blocks(index_tip)
            and not self.camera.counting_down
            and not self.stylus.contains(index_tip)
            and not self.palette.contains(index_tip)
            and not (self.outline.active and self.trace_controls.contains(index_tip))
        )
        if can_paint and self.eraser.active:
            self.canvas.lift_pen()
            self.eraser.erase(self.canvas, index_tip)
        elif can_paint:
            self.eraser.lift()
            self.canvas.draw_line(index_tip, self.palette.selected_color, self.stroke_profile())
        else:
            self.canvas.lift_pen()
            self.eraser.lift()
        for layer in self.canvas.layers():
            self.zone.clip(layer)
            self.safe_zones.clip(layer)

    def pause_drawing(self):
        # the artwork stays on the canvas; only in-progress strokes, countdowns and hovers are dropped
        self.canvas.lift_pen()
        self.eraser.lift()
        self.selector.reset()
        self.camera.countdown_start = None
        self.clearer.state = ScreenClearer.IDLE
        self.info.hovered = False
        self.stylus.close()
        self.palette.close()

    def open_instructions(self):
        self.pause_drawing()
        self.welcome.restart()

    #the selected stylus at the selected size; while tracing the size scales the outline's fill width instead
    def stroke_profile(self):
        if self.outline.active:
            return self.outline.fill_profile(self.stylus.selected, self.stylus.size_factor)
        return self.stylus.current

    def open_trace(self):
        self.pause_drawing()
        self.trace_menu.open()

    #starts over: the drawing and the outline are wiped and the picker comes back
    def change_icon(self):
        self.canvas.clear()
        self.outline.clear()
        self.open_trace()

    #carries out an emoji pick, a typed name or Cancel from the overlay
    def apply_trace(self, action):
        if action is None:
            return
        kind, value = action
        if kind == "emoji":
            self.outline.set_emoji(value)
        elif kind == "submit":
            self.outline.set_text(value)
        self.trace_menu.close()

    def handle_gesture(self, gesture, points):
        index_tip = points[HandTracker.INDEX_TIP] if points else None
        if self.info.update(gesture, index_tip):
            self.open_instructions()
            return
        self.clearer.update(gesture == "fist", self.canvas)
        self.handle_toolbar(gesture, index_tip)
        self.handle_painting(gesture, index_tip)

    def draw_cursor(self, frame, gesture, points):
        if not points:
            return
        index_tip = points[HandTracker.INDEX_TIP]
        if gesture == "draw" and self.eraser.active:
            self.eraser.draw_cursor(frame, index_tip)
        elif gesture == "draw":
            radius = max(self.stroke_profile().thickness // 2, 5)
            cv2.circle(frame, index_tip, radius, self.palette.selected_color, cv2.FILLED)
        elif gesture == "cursor":
            self.cursor.draw(frame, index_tip)

    def render_drawing(self, frame, points, gesture):
        self.handle_gesture(gesture, points)

        self.outline.draw(frame)
        artwork = self.canvas.overlay(frame)
        self.zone.draw(artwork)
        saved = self.camera.capture_if_ready(artwork)
        if saved:
            print(f"Saved screenshot to {saved}")

        display = artwork.copy()
        if points:
            self.tracker.draw_landmarks(display, points)
        self.safe_zones.draw(display)
        self.edge_warning.draw(display, self.zone)
        self.palette.draw(display, self.eraser.active)
        self.eraser.draw(display)
        self.stylus.draw(display, self.palette.selected_color, self.outline.fill_thickness)
        if not self.palette.is_open:
            self.stylus.draw_caption(display, self.stroke_profile())
        self.selector.draw(display, points[HandTracker.INDEX_TIP] if points else None)
        self.camera.draw(display)
        if self.outline.active:
            self.trace_controls.draw(display)
        else:
            self.trace_button.draw(display)
        self.info.draw(display)
        self.clearer.draw(display)
        self.status.draw(display, self.palette, self.eraser, gesture, self.stylus)
        self.draw_cursor(display, gesture, points)
        return display

    def render_trace(self, frame, points, gesture):
        index_tip = points[HandTracker.INDEX_TIP] if points else None
        self.apply_trace(self.trace_menu.update(gesture, index_tip))
        self.outline.draw(frame)
        display = self.canvas.overlay(frame)
        self.trace_menu.draw(display, index_tip)
        if points:
            self.tracker.draw_landmarks(display, points)
        if gesture == "cursor":
            self.cursor.draw(display, index_tip)
        return display

    def render_welcome(self, frame, points, gesture):
        self.welcome.update(gesture)
        return self.welcome.render(frame, points, gesture, self.tracker)

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
        try:
            while True:
                frame = self.read_frame()
                if frame is None:
                    break

                points = self.tracker.find_landmarks(frame)
                points, gesture = self.tracking_box.update(points, self.gestures.classify(points))

                if not self.welcome.complete:
                    display = self.render_welcome(frame, points, gesture)
                elif self.trace_menu.is_open:
                    display = self.render_trace(frame, points, gesture)
                else:
                    display = self.render_drawing(frame, points, gesture)

                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                # while the overlay is open the keyboard types into it, so q and Esc don't quit
                if self.trace_menu.is_open:
                    self.apply_trace(self.trace_menu.handle_key(key))
                elif key in (ord("q"), 27):
                    break
        finally:
            self.capture.release()
            self.tracker.close()
            cv2.destroyAllWindows()


#entry point for the application
if __name__ == "__main__":
    AirCanvasApp().run()

