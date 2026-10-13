"""Real-time AI finger counter using MediaPipe Tasks and OpenCV."""
from pathlib import Path
import math
import time
import urllib.request

import cv2
import mediapipe as mp

MODEL = Path(__file__).with_name("hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

def ensure_model():
    if not MODEL.exists() or MODEL.stat().st_size < 100_000:
        print("Downloading MediaPipe hand model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL)

def angle(a,b,c):
    u=(a.x-b.x,a.y-b.y)
    v=(c.x-b.x,c.y-b.y)
    dot=u[0]*v[0]+u[1]*v[1]
    norm=math.hypot(*u)*math.hypot(*v)
    return math.degrees(math.acos(max(-1,min(1,dot/norm)))) if norm else 0

def count_fingers(p):
    # Both joints should be reasonably straight; works for either hand.
    chains=[(1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]
    states=[]
    for base,middle,near_tip,tip in chains:
        threshold=150 if base==1 else 155
        states.append(angle(p[base],p[middle],p[near_tip])>threshold and angle(p[middle],p[near_tip],p[tip])>threshold)
    return sum(states)

def main():
    ensure_model()
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options=vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6)
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check Windows camera permissions and close other camera apps.")
    previous=time.perf_counter()
    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            start=time.monotonic()
            last_timestamp=-1
            while True:
                ok,frame=cap.read()
                if not ok:
                    print("Camera frame unavailable")
                    break
                frame=cv2.flip(frame,1)
                h,w=frame.shape[:2]
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                timestamp=max(last_timestamp+1,int((time.monotonic()-start)*1000))
                last_timestamp=timestamp
                result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),timestamp)
                total=0
                for hand in result.hand_landmarks:
                    total+=count_fingers(hand)
                    for a,b in CONNECTIONS:
                        pa=(int(hand[a].x*w),int(hand[a].y*h))
                        pb=(int(hand[b].x*w),int(hand[b].y*h))
                        cv2.line(frame,pa,pb,(80,230,80),2)
                    for point in hand:
                        cv2.circle(frame,(int(point.x*w),int(point.y*h)),4,(0,180,255),-1)
                now=time.perf_counter()
                fps=1/max(now-previous,0.001)
                previous=now
                cv2.rectangle(frame,(0,0),(390,100),(30,30,30),-1)
                cv2.putText(frame,f"Fingers: {total}",(15,48),cv2.FONT_HERSHEY_SIMPLEX,1.25,(0,255,120),3)
                cv2.putText(frame,f"Hands: {len(result.hand_landmarks)}  FPS: {fps:.1f}  Q: quit",(15,83),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
                cv2.imshow("AI Finger Counter",frame)
                if cv2.waitKey(1)&0xFF==ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
