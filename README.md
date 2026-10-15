# Air Canvas

> A touchless drawing tool that lets you paint in the air using just your webcam and hand movements.

![Air Canvas](docs/demo.JPG)

## About
Air Canvas turns your hand into a stylus. A webcam tracks your fingers in real time, and different hand gestures switch between drawing, selecting colors, and clearing the canvas, with no mouse or touchscreen needed. I built it as my first prompt engineering project, to learn how to turn an idea into a working app by writing clear, iterative prompts.

## How I built it
I built v1 with Claude Code in 12 prompts:
- **Structured prompts:** each prompt followed a Role → Task → Context → Constraints format.
- **Iterative prompting:** I tested every version, noted what worked and what didn't, and used that to write the next prompt.
- **Code review:** after v1 was done, I went through the code line by line and wrote the comments to understand the code. 

See every prompt in [prompts/PROMPTS.md](prompts/PROMPTS.md).

## Features
| Mode | Gesture | What it does |
| --- | --- | --- |
| Draw Mode | ☝️ Index finger up | Draw on the canvas (or erase, when the eraser is selected) |
| Cursor Mode | ✌️ Index + middle fingers up | Move around without drawing; hover over a color or the eraser to select it |
| Clear Mode | ✊ Closed fist | Clear the whole canvas after a 3-2-1 countdown (open your hand to cancel) |

Also includes:
- 6 colors and an eraser in the top toolbar
- Screenshot button: point at the camera icon to save your drawing to Downloads
- Info button: reopens the instructions screen
- Shaded no-draw zones and a flashing warning near the edges
- Welcome screen with hand calibration

## Getting started
### Requirements
- python3 --version
- A webcam

### Install and run
```bash
git clone https://github.com/aineshkumar7official/air-canvas.git
cd air-canvas
pip install -r requirements.txt
python3 airCanvasv1.py
```
The hand-tracking model downloads automatically the first time you run it. Press **q** or **Esc** to quit.

## Tech stack
- **Python**
- **OpenCV:** webcam input, drawing, and the app window
- **MediaPipe:** tracks 21 points on your hand
- **NumPy:** stores images as arrays
- **Pillow:** renders the custom fonts

## Roadmap
See [ROADMAP.md](ROADMAP.md)

## Author
**Ainesh Kumar**
USF | B.S. Computer Science & Business, FinTech | Class of 2030
[LinkedIn](https://www.linkedin.com/in/aineshkumarofficial/) · [GitHub](https://github.com/aineshkumar7official)

