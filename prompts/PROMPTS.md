# Prompt Log

Air Canvas v1 was built through the prompts below using Claude Code. Each prompt is shown exactly as I wrote it, followed by my reflection on what changed.

> **My prompt structure:** Role → Task → Context → Constraints

## Overview

| # | Focus |
| --- | --- |
| [1](#prompt-1-core-drawing-app) | Core drawing app |
| [2](#prompt-2-eraser-button-camera-placement-and-canvas-border) | Eraser button, camera placement, and canvas border |
| [3](#prompt-3-cursor-mode) | Cursor mode |
| [4](#prompt-4-safe-zones-selection-animation-and-eraser-brush) | Safe zones, selection animation, and eraser brush |
| [5](#prompt-5-fist-clear-countdown-and-shaded-zones) | Fist-clear countdown and shaded zones |
| [6](#prompt-6-welcome-and-calibration-screen) | Welcome and calibration screen |
| [7](#prompt-7-custom-ui-design-icons-and-fonts) | Custom UI design (icons and fonts) |
| [8](#prompt-8-adaptive-edge-warning) | Adaptive edge warning |
| [9](#prompt-9-single-border-fix) | Single border fix |
| [10](#prompt-10-shaded-no-drawing-zones) | Shaded no-drawing zones |
| [11](#prompt-11-expanded-hand-tracking) | Expanded hand tracking |
| [12a](#prompt-12a-information-box-first-draft) | Information box (first draft) |
| [12b](#prompt-12b-information-box-navigation) | Information box navigation |

## Overall reflection

From this project, I learned how to use different prompt engineering techniques to guide an AI assistant through iterative prompting which involved testing each version, describing what worked and didn't. I understood what detailed and organized prompts produce better results than short ones and its important to check code and ensure every requirements get built.

---

## Prompt 1: Core drawing app

```text
Role 
Act as an expert Python and computer vision engineer. 

Task 
I want to build a Python-based art application that uses my computer's webcam to let users draw in the air with hand gestures and take photos of their creations.

Context 

How the drawing works
1.	Libraries to use: OpenCV and MediaPipe for hand tracking and computer vision
2.	Resolution: Design the system assuming a 1280 x 720 video feed

Drawing Mode
1.	When the user holds up the index finger, let the user draw a line for tracking.
2.	Eraser Toggle: When the index finger and thumb touch each other, turn on eraser mode. When they touch each other again, re-enable the drawing mode
3.	Clear Screen:  When the user holds up a fist, clear the whole canvas.

Where to get colors
1.	At the top, place boxes with text (blue, green, red, purple, yellow)
2.	Color Selection: When the user holds up the index/middle fingers within the boundary of the box. The color is then selected. 

How to take a screenshot
1.	Place the camera icon in the bottom right corner
2.	When the user points the index finger at the camera for more than one second.  Start a 5-second timer
3.	Take a screenshot as a PNG and save it to the user’s Downloads folder.  

Constraints
1.	In the code, separate out each of the features
2.	Don’t write comments; I will add them to show I understand the code. 

Place the code in Visual Studio Code IDE. 
```

**What changed:** The first build set up the UI which included buttons, drawing and screenshots well. However, the pinch-to-erase toggle did not work.

---

## Prompt 2: Eraser button, camera placement, and canvas border

```text
Okay, the code so far handles drawing and color selection well. Here is what we need to improve on.

Eraser Mode
1.	Eraser specifc section: Switch from pinch to erase to hover over the eraser box to trigger the eraser. 
2.	Clear screen: When the user holds up a fist, clear the whole canvas

Screenshot mode
1.	UI Placement: Place the camera icon overlay in the top-right corner of the video feed. 
2.	Click Logic: Update finger-tip collision logic so the user can press specifc area to trigger a canvas screen. 

Canvas
1.	Drawing boundaries: Put a black border around where users can draw within the canvas zone
2.	Input Restriction: Prevent the user from drawing outside the input area.

Please update the existing code structure to implement these changes cleanly. 
```

**What changed:** The pinch gesture was replaced with an eraser button you hover over to stop accidental triggers. The clear-all and screenshot features worked more reliably and the canvas was more limited.

---

## Prompt 3: Cursor mode

```text
Okay, the camera button and UI are working well, but we need to establish one element. 
1.	Cursor Mode: When the user holds up just the middle and index fingers, 
2.	How it works: No drawing or erasing should occur. 
3.	Visual Feedback: Have a small neutral white dot so the user knows where they are going. 
```

**What changed:** The app had a cursor mode where user held up index finger and middle finger which allowed user to preview where they'll draw before they start.

---

## Prompt 4: Safe zones, selection animation, and eraser brush

```text
Okay, cursor mode looks good. Now need to do a few refinements

1.	Hot corners/sections (Canvas Safe Zones): Define explicit boundary areas where drawing lines is disabled 
2.	Color/Eraser Selection: When the user hovers over a color box or eraser box, render a visual animation to confirm selection. 
3.	Eraser Mode Mechanics: When Eraser mode is active, the index finger tracks a localized eraser brush that clears specifc pixels, not the whole thing. 
```

**What changed:** The UX animations now confirms when a color is picked, safe zones are more explicit and the eraser mode only cleared what the index finger wanted to clear instead of whole canvas.

---

## Prompt 5: Fist-clear countdown and shaded zones

```text
Okay, the eraser tool works well, and animations are good. A few more refinements

Safe Zones: 
1.	When a fist gesture is detected, do not clear the screen instantly. Instead, start a visible 3-2-1 countdown timer on screen. If the user opens their hand/breaks the fist gesture before the countdown reaches 0, cancel the timer and preserve the canvas. Only clear the canvas if the fist is held continuously until the countdown finishes.

•	Explicit Safe Zones (Top and Bottom): Make the non-drawing menu areas highly distinct. Render a solid, clean boundary line or a semi-transparent background bar across both the top menu zone and the bottom interface zone so it is immediately obvious to the user where they are allowed to draw and where drawing is disabled.
For the fist gesture/clearning screen. When the user makes a fist and the screen clears, put a message: " Screen cleared and let the user continue drawing. If the user still holds up a fist, then show the clear message, then redo the countdown. 

Fist Clear Loop Logic:
- Stage 1 (Countdown): When a fist is detected, run the 3-2-1 countdown. If the fist is broken, reset.
- Stage 2 (Clear & Notify): When the countdown hits 0, clear the canvas and instantly flash a "Screen Cleared!" text message on the screen.
- Stage 3 (Continuous Hold Check): 
  - If the user keeps holding the fist up after the clear message appears, wait 1.5 seconds, remove the message, and automatically start a brand new 3-2-1 countdown loop.
  - If the user opens their hand at any point after the clear, remove the notification text immediately and return to normal drawing mode.
```

**What changed:** Instead of clearing the canvas instantly, there was a 3-2-1 countdown giving users time to cancel and a notifcation so users knows when its okay to draw again.

---

## Prompt 6: Welcome and calibration screen

```text
The drawing, eraser, and cursor modes work well. 

Now I want to have an instructions and calibration page that appears before the user enters the main drawing canvas. 

Please implement the following layout, UI elements, and transition logic.

1.	State Management:
1.	Create an initial state/screen. Drawing only begins when calibration is successfully completed. 

2.	UI Layout & Typography (Centered on screen)
1.	Title: “Welcome to Air Canvas”
2.	Subheading: “Drawing redefined”
3.	Footer instruction: (press q to quit at any time)

3.	Visual Guide Panels
1.	Panel 1 (Draw Mode): Text: "1. Use your index finger to draw." (Visual: Index finger up, others down).
2.	Panel 2 (Cursor/Hover Mode): Text: "2. Raise index + middle fingers to hover/select colors." (Visual: Two fingers up).
3.	Panel 3 (Clear Screen): Text: "3. Make a closed fist to clear the canvas." (Visual: Closed fist).
4.	Panel 4 (Snapshot Mode): Text: "4. Hover over the camera icon to save a screenshot." (Visual: Camera icon graphic).

4. Calibration & Transition Logic: - Call-to-Action Text: 

1.	Display a blinking or highlighted prompt: "(Hold up just your index finger to calibrate and start)" - 
2.	Detection Check: The system must actively scan the webcam feed while on this page. If it detects the user successfully holding up exactly their index finger (and no other fingers) for a sustained 2 seconds (e.g., a simple frame counter), trigger the transition.
```

**What changed:** Before the user did not know what to do when the canvas opened, now there was a clear welcome screen for how to start.

---

## Prompt 7: Custom UI design (icons and fonts)

```text
Okay, I want to focus on the UI elements right now.

Look at the design elements in the AirCanvas interface and AirCanvas fonts and use those images/fonts to update the UI. 
```

**What changed:** The default UI involved plain hard to rad boxes so I designed custom icons and fonts to make it clearer.

---

## Prompt 8: Adaptive edge warning

```text
 Now, I want to implement an adaptive "Safe Zone" and "Danger Zone" boundary system on the drawing canvas to prevent users from drawing too close to the edges or UI elements. Please update the canvas logic with the following adaptive boundary rules: 

1. Boundary Definitions: - Safe Zone: 
The central area of the canvas where drawing is fully enabled. - 
Warning/Danger Zone: A buffer zone (e.g., 40-50 pixels wide) tracking along the inside of the black canvas border and around the camera button UI area. 

2. Adaptive Visual Feedback (Border Color Change): - 
Normal State: While the user's active drawing fingertip is well within the Safe Zone, keep the canvas border its default color (black). - Warning State: If the user's fingertip moves into the Warning/Danger Zone buffer, dynamically change the canvas border color from black to flashing red. - 
On-Screen Warning: When in the Danger Zone, overlay a small, semi-transparent text warning near the top of the screen that says "WARNING: Approaching Edge".

Please integrate this adaptive color-changing border and warning logic cleanly into our existing coordinates check loop.
```

**What changed:** The user kept drawing in areas where they couldn't draw. Now they were alerted ahead of time with a flashing border  to avoid the danger areas.

---

## Prompt 9: Single border fix

```text
Okay, the color-changing border works well, but the UI needs to be improved

Current state
1.	Massive black border, then inside a yellow border

What to improve
1.	Merge into one black border around
2.	 If the user's fingertip moves into the Warning/Danger Zone buffer, dynamically change the canvas border color from black to flashing red. - 
3.	On-Screen Warning: When in the Danger Zone, overlay a small, semi-transparent text warning near the top of the screen that says "WARNING: Approaching Edge".
Please integrate this adaptive color-changing border and warning logic cleanly into our existing coordinates check loop.
```

**What changed:** Before the canvas was small because the border was not clear. The result was a single dynmaic black border which flashes red when user moves finger near the edges.

---

## Prompt 10: Shaded no-drawing zones

```text
Okay, the black border looks much better. There are still some UI’s that need to be improved.

1.	No drawing zones: put shaded zones showcasing where user cannot draw 
2.	When user enters that zone trigger warning system as well. 

The black canvas border looks much better. Now, I want to visually define the "No-Drawing Zones" (such as the top UI header area or any edge margins) and connect them to our warning system. 

Please update the UI and boundary logic with the following rules:

1. Visual Shaded Zones:
- Create a clear visual distinction for all "No-Drawing Zones" (e.g., the top menu bar where the camera icon sits). 
- Render these restricted zones using a semi-transparent colored overlay (e.g., a gray or red tint with 30% opacity) so the user can still see the webcam feed underneath, but clearly knows it is a blocked area.

2. Warning System Trigger:
- Monitor the coordinates of the user's active tracking fingertip.
- Instant Trigger: The moment the user's fingertip enters the boundaries of any shaded "No-Drawing Zone", immediately trigger the canvas warning system.
- Visual Feedback: Change the main canvas border to solid red, display an on-screen text warning ("Restricted Area - Drawing Disabled"), and automatically pause any line-drawing or point-appending inputs.

Please integrate these transparent shaded blocks and coordinate checks cleanly into our main drawing loop.
```

**What changed:** The black border did not include restricted areas and did not include all the places where the user couldn't draw. Now it was explict about where user can or cannot draw.

---

## Prompt 11: Expanded hand tracking

```text
The UI zones and warnings are working well. However, there is an issue with hand tracking because the box is not wide enough. 

Please update the following requirements. 

1. Expand the Active Tracking Box:
- Increase the overall dimensions of the active hand-tracking bounding box. It needs to be significantly larger to comfortably fit a person's entire open hand with extended fingers without clipping out.

2. Dynamic/Adaptive Padding:
- Add a generous pixel padding/margin (e.g., 50–80 pixels) around the detected hand landmarks. 
- Ensure that even if the user moves quickly or opens their hand fully, the tracking box expands dynamically to keep all 21 MediaPipe hand landmarks completely inside the active tracking field.

3. Edge-Tolerance Logic:
- If the hand approaches the outer boundaries of this expanded box, gracefully maintain tracking rather than instantly dropping the hand state or misinterpreting gestures.
```

**What changed:** User would place whole hand but it would not start tracking.The prompt created a bigger tracking box so hand tracking was clearer.

---

## Prompt 12a: Information box (first draft)

```text
Okay, the final task of this version. 

Look at the design elements in the AirCanvas interface and create an information box where the user can hover to remember the instructions.

How to enter the instruction page
1.	Hold up 2 fingers 

To exit
1.	Recalibrate hand. 

The expanded tracking box is working perfectly. This is the final task for this version of the application. 
```

**What changed:** [Add your reflection for this prompt]

---

## Prompt 12b: Information box navigation

```text
I want to add a persistent "Information Box" UI element to the main drawing canvas. This will act as a quick-reference guide so users can easily recall how to navigate back to the instructions/calibration page.

Please implement the following UI design and navigation logic:

1. Information Box UI Design:
Review the design elements in the AirCanvas interface and create an information box that users can hover over to recall the instructions.

2. Navigation Logic (Re-entering the Instructions Page):
- Activation Gesture: While on the main drawing canvas, if the user holds up exactly 2 fingers (Index + Middle) over the Information Box trigger area, instantly pause the drawing canvas state.
- Action: Redirect the user back to the main "Instructions and Calibration" landing page that we built earlier.

3. Exit Logic (Returning to Canvas):
- Recalibration Trigger: Once the user is back on the Instructions page, they must perform the full hand recalibration routine (holding up just the index finger for 2 seconds) to exit the instructions page and return to their active canvas.

Please integrate this final Information UI component and its state-switching logic cleanly into our main project loop.
```

**What changed:** Added a information button so user could hover on over if they forgot the instructions and recalibrate their hand.
