# Hand Gesture Desktop Controller

A native Python desktop application that turns your standard webcam and hand movements into a precision mouse controller with instant response and clean gestures.

---

## 🛠 Hand Gestures

* **👌 Pinch & Move (0ms Delay)**: **Move Mouse**
  * Pinch your forefinger (index finger) and thumb together to immediately engage and move the cursor smoothly in real time.
  * Release your fingers to freeze the cursor in place.

* **👆 Quick Pinch Tap**: **Left Click**
  * A quick tap between forefinger and thumb (pinch and release within 0.35s) executes an instant **Left Click**.
  * A visual ripple animation appears on screen to confirm the click.

* **⚡ Double Pinch Tap**: **Double Click**
  * Two quick pinch taps in succession execute a **Double Click**.

* **👉 Thumb + Middle Finger Pinch**: **Right Click**
  * Pinch thumb and middle finger together.

* **✌️ Two-Finger Vertical Move**: **Scroll Up / Down**
  * Extend index and middle fingers together and move up/down to scroll.

---

## ⚙️ Precision & Active Region

* **Enlarged Active Canvas (88% × 84%)**: Generous room across your webcam frame to reach all corners of your screen.
* **Dynamic Box Resizing**: Press `[+]` or `[-]` to expand or shrink the active region.
* **Calibrated One-Euro Filter**: Jitter-free, steady cursor positioning.
* **Realistic Thresholds (`0.18` / `0.28`)**: Natural finger touching without straining.

---

## ⌨️ Hotkeys & Controls

| Key | Action |
|---|---|
| **`[SPACE]`** | Pause / Resume mouse control |
| **`[M]`** | Cycle Modes: Pinch-to-Move $\rightarrow$ Relative Clutch $\rightarrow$ Continuous Pointing |
| **`[+]` / `[-]`** | Expand / Shrink active screen region |
| **`[C]`** | Start interactive 2-step calibration wizard |
| **`[ESC]` / `[Q]`** | Exit application |
| *Corner Slam* | Hardware failsafe: slam cursor into any corner to pause |

---

## 🚀 Running the Application

In your terminal or by double-clicking `run.bat`:
```bash
cd C:\Users\moame\hand_control
.venv\Scripts\activate
python main.py
```

## 🧪 Running Tests
```bash
python -m unittest test_modules.py
```
