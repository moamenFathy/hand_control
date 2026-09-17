# Hand Gesture Desktop Controller

A native Python desktop application that turns your standard webcam and hand movements into a precision mouse controller with unambiguous, robust gesture triggers.

---

## 🛠 Hand Gestures

* **⏳ Hold Forefinger + Thumb for 2 Seconds**: **Unlock & Move Mouse**
  * Pinch and hold your forefinger and thumb together for **2 seconds**.
  * A circular countdown ring fills around your fingers on screen.
  * Once 2.0s is reached, the cursor **unlocks and moves** smoothly as you move your hand.
  * Release your fingers to instantly lock the cursor in place.

* **👆 Double-Tap Pinch**: **Left Click**
  * Perform **2 quick pinch taps** in succession (pinch $\rightarrow$ release $\rightarrow$ pinch $\rightarrow$ release within ~0.5s) to execute a Left Click.
  * Single accidental pinch taps will **not** trigger unwanted clicks.

* **👉 Thumb + Middle Finger Pinch**: **Right Click**
  * Pinch thumb and middle finger together.

* **✌️ Two-Finger Vertical Move**: **Scroll Up / Down**
  * Extend index and middle fingers together and move up/down to scroll.

---

## ⚙️ Precision & Active Region

* **Enlarged Active Canvas (88% × 84%)**: Ample room across your webcam frame to reach all corners of your display effortlessly.
* **Live Box Resizing**: Press `[+]` or `[-]` to dynamically expand or shrink the active region.
* **Calibrated One-Euro Filter**: Steady, jitter-free cursor tracking with zero hand-tremor wobble.

---

## ⌨️ Hotkeys & Controls

| Key | Action |
|---|---|
| **`[SPACE]`** | Pause / Resume mouse control |
| **`[M]`** | Cycle Modes: Hold 2s to Move $\rightarrow$ Relative Clutch $\rightarrow$ Continuous |
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
