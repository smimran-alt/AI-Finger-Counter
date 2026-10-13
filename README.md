# AI Finger Counter

A webcam-based real-time Python finger counter using OpenCV and MediaPipe Tasks. Detects up to two hands, draws hand landmarks, and estimates 0–10 raised fingers.

## Run on Windows with Python 3.12

Open this repository in VS Code and run:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 finger_counter.py
```

On first run, the official MediaPipe hand landmark model is downloaded. Allow camera access, show your fingers, and press **Q** to quit.

**Notes:** Finger counting uses joint-angle rules and may miscount bent, overlapping, or rotated fingers. Processing happens locally. GitHub Pages cannot run this desktop webcam Python program.
