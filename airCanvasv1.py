# distance between two points
import math
# standard libraries for time and URL handling
import time
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
    TEXT.draw(frame, text, ((x1 + x2) // 2, (y1 + y2) // 2), "bold", 22, text_color, stroke=2, centered=True)


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

    #draws the icon on the frame at the specified box location
    def draw(self, frame, name, box):
        x1, y1, x2, y2 = box
        icon = self.fitted(name, x2 - x1, y2 - y1)
        height, width = icon.shape[:2]
        blend(frame, icon, x1 + (x2 - x1 - width) // 2, y1 + (y2 - y1 - height) // 2)


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


#color palette for selecting drawing colors
class ColorPalette:
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
        for i, (name, color) in enumerate(self.colors):
            x1 = TOOLBAR_LEFT + i * (BUTTON_WIDTH + BUTTON_GAP)
            box = (x1, TOOLBAR_TOP, x1 + BUTTON_WIDTH, TOOLBAR_TOP + TOOLBAR_HEIGHT)
            self.boxes.append((name, color, box))
        self.selected_name, self.selected_color = self.colors[0]

    @property
    def right_edge(self):
        return self.boxes[-1][2][2]

    def box_at(self, point):
        for name, color, box in self.boxes:
            if point_in_box(point, box):
                return name, box
        return None

    def select(self, name):
        for color_name, color, box in self.boxes:
            if color_name == name:
                self.selected_name, self.selected_color = color_name, color

    def draw(self, frame, eraser_active):
        for name, color, box in self.boxes:
            x1, y1, x2, y2 = box
            highlighted = name == self.selected_name and not eraser_active
            if self.icons.has(name):
                self.icons.draw(frame, name, box)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), 2)
                draw_label(frame, name, box)
            if highlighted:
                draw_selection(frame, box)


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
            TEXT.draw(frame, "i", center, "bold", 48, centered=True)
        cv2.circle(frame, center, radius, (255, 255, 255), 2)
        if self.hovered:
            draw_selection(frame, self.box)
            TEXT.draw(frame, self.HINT, (center[0], y2 + 22), "semibold", 14, stroke=2, centered=True)


#keeps track of user behavior
class StatusDisplay:
    def draw(self, frame, palette, eraser, gesture):
        mode = "ERASER" if eraser.active else f"DRAW ({palette.selected_name})"
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
        self.eraser = Eraser(self.palette.right_edge + BUTTON_GAP, self.icons)
        self.zone = DrawingZone(FRAME_WIDTH, FRAME_HEIGHT)
        self.safe_zones = SafeZones(FRAME_WIDTH, FRAME_HEIGHT, self.zone)
        self.edge_warning = EdgeWarning(self.zone, self.safe_zones)
        self.selector = HoverSelector()
        self.canvas = DrawingCanvas(FRAME_WIDTH, FRAME_HEIGHT)
        self.clearer = ScreenClearer()
        self.camera = ScreenshotCamera(FRAME_WIDTH, DOWNLOADS_DIR, self.icons)
        self.info = InfoButton(self.camera.box[0] - BUTTON_GAP, self.icons)
        self.cursor = CursorMode()
        self.status = StatusDisplay()
        self.welcome = WelcomeScreen(self.icons)

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
            hit = self.palette.box_at(index_tip)
            if hit:
                name, box = hit
                return ("color", name), box
        if gesture in ("draw", "cursor") and self.eraser.contains(index_tip):
            return ("eraser", None), self.eraser.box
        return None, None

    def handle_toolbar(self, gesture, index_tip):
        pointing = gesture in ("draw", "cursor")
        self.camera.update_press(index_tip if pointing else None)
        target, box = self.find_toolbar_target(gesture, index_tip)
        confirmed = self.selector.update(target, box)
        if confirmed is None:
            return
        kind, name = confirmed
        if kind == "color":
            self.palette.select(name)
            self.eraser.active = False
        else:
            self.eraser.active = True

    def handle_painting(self, gesture, index_tip):
        on_button = (
            self.palette.box_at(index_tip) is not None
            or self.eraser.contains(index_tip)
            or self.camera.contains(index_tip)
            or self.info.contains(index_tip)
        )
        self.edge_warning.update(gesture, index_tip, on_button)
        can_paint = (
            gesture == "draw"
            and self.zone.contains(index_tip)
            and not self.edge_warning.restricted
            and not self.safe_zones.blocks(index_tip)
            and not self.camera.counting_down
        )
        if can_paint and self.eraser.active:
            self.canvas.lift_pen()
            self.eraser.erase(self.canvas, index_tip)
        elif can_paint:
            self.eraser.lift()
            self.canvas.draw_line(index_tip, self.palette.selected_color)
        else:
            self.canvas.lift_pen()
            self.eraser.lift()
        self.zone.clip(self.canvas.canvas)
        self.safe_zones.clip(self.canvas.canvas)

    def open_instructions(self):
        # the artwork stays on the canvas; only in-progress strokes, countdowns and hovers are dropped
        self.canvas.lift_pen()
        self.eraser.lift()
        self.selector.reset()
        self.camera.countdown_start = None
        self.clearer.state = ScreenClearer.IDLE
        self.info.hovered = False
        self.welcome.restart()

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
            cv2.circle(frame, index_tip, 10, self.palette.selected_color, cv2.FILLED)
        elif gesture == "cursor":
            self.cursor.draw(frame, index_tip)

    def render_drawing(self, frame, points, gesture):
        self.handle_gesture(gesture, points)

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
        self.selector.draw(display, points[HandTracker.INDEX_TIP] if points else None)
        self.camera.draw(display)
        self.info.draw(display)
        self.clearer.draw(display)
        self.status.draw(display, self.palette, self.eraser, gesture)
        self.draw_cursor(display, gesture, points)
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

                if self.welcome.complete:
                    display = self.render_drawing(frame, points, gesture)
                else:
                    display = self.render_welcome(frame, points, gesture)

                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            self.capture.release()
            self.tracker.close()
            cv2.destroyAllWindows()


#entry point for the application
if __name__ == "__main__":
    AirCanvasApp().run()

