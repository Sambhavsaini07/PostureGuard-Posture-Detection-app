from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import uvicorn
from pydantic import BaseModel
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="PostureGuard Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://posture-guard-posture-detection-ky6ao0j89.vercel.app",
        "http://localhost:3000",  # For local development
        "https://localhost:3000"  # For local HTTPS development
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Connected clients
connected_clients = set()

# Data models
class PostureAnalysis(BaseModel):
    score: int
    status: str
    angles: Dict[str, float]
    feedback: List[str]
    timestamp: str
    keypoints: Optional[Dict] = None

class PostureRules:
    """Rule-based posture analysis logic"""
    
    @staticmethod
    def calculate_angle(point1, point2, point3):
        """Calculate angle between three points"""
        try:
            # Convert to numpy arrays
            p1 = np.array([point1.x, point1.y])
            p2 = np.array([point2.x, point2.y])
            p3 = np.array([point3.x, point3.y])
            
            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return math.degrees(angle)
        except:
            return 0.0
    
    @staticmethod
    def calculate_neck_angle(nose, left_ear, right_ear, left_shoulder, right_shoulder):
        """Calculate neck forward angle - improved version"""
        try:
            # Calculate average ear position
            avg_ear_x = (left_ear.x + right_ear.x) / 2
            avg_ear_y = (left_ear.y + right_ear.y) / 2
            
            # Calculate average shoulder position
            avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Calculate the angle between ear-shoulder line and vertical
            # Vertical line goes downward from ear position
            vertical_point = type('Point', (), {'x': avg_ear_x, 'y': avg_ear_y + 0.1})()
            ear_point = type('Point', (), {'x': avg_ear_x, 'y': avg_ear_y})()
            shoulder_point = type('Point', (), {'x': avg_shoulder_x, 'y': avg_shoulder_y})()
            
            # Calculate angle - this represents forward head posture
            angle = PostureRules.calculate_angle(vertical_point, ear_point, shoulder_point)
            
            # Convert to a more intuitive measure (0 = perfect, higher = worse)
            neck_forward_angle = abs(90 - angle)
            
            return neck_forward_angle
        except Exception as e:
            logger.error(f"Error calculating neck angle: {e}")
            return 0.0
    
    @staticmethod
    def calculate_back_angle(left_shoulder, right_shoulder, left_hip, right_hip):
        """Calculate back angle (torso alignment)"""
        try:
            # Calculate average shoulder and hip positions
            avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            
            avg_hip_x = (left_hip.x + right_hip.x) / 2
            avg_hip_y = (left_hip.y + right_hip.y) / 2
            
            # Calculate angle between shoulder-hip line and vertical
            vertical_point = type('Point', (), {'x': avg_shoulder_x, 'y': avg_shoulder_y - 0.1})()
            shoulder_point = type('Point', (), {'x': avg_shoulder_x, 'y': avg_shoulder_y})()
            hip_point = type('Point', (), {'x': avg_hip_x, 'y': avg_hip_y})()
            
            angle = PostureRules.calculate_angle(vertical_point, shoulder_point, hip_point)
            
            # For back angle, we want close to 180 degrees for straight posture
            return angle
        except Exception as e:
            logger.error(f"Error calculating back angle: {e}")
            return 180.0

    @staticmethod
    def analyze_sitting_posture(landmarks) -> PostureAnalysis:
        """Analyze sitting posture based on landmarks"""
        feedback = []
        score = 100
        status = "good"
        
        try:
            # Extract key landmarks
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            
             # Calculate angles with improved methods
            neck_angle = PostureRules.calculate_neck_angle(
                nose, left_ear, right_ear, left_shoulder, right_shoulder
            )
            back_angle = PostureRules.calculate_back_angle(
                left_shoulder, right_shoulder, left_hip, right_hip
            )
            
            # Calculate knee angle (using hip, knee, and ankle)
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            knee_angle = PostureRules.calculate_angle(left_hip, left_knee, left_ankle)
            
            # Shoulder level check
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            
            # Head tilt check
            head_tilt = abs(left_ear.y - right_ear.y)
            
            # Apply rules
            if neck_angle > 30:
                feedback.append("⚠️ Neck is too forward - straighten your neck")
                score -= 15
                status = "warning"
            
            if back_angle < 150:
                feedback.append("⚠️ Back is hunched - sit up straight")
                score -= 20
                status = "bad"
            
            if knee_angle < 90:
                feedback.append("⚠️ Knee position may cause strain")
                score -= 10
                if status == "good":
                    status = "warning"
            
            if shoulder_diff > 0.05:
                feedback.append("⚠️ Shoulders are uneven - balance your posture")
                score -= 10
                if status == "good":
                    status = "warning"
            
            if head_tilt > 0.03:
                feedback.append("⚠️ Head is tilted - keep your head level")
                score -= 8
                if status == "good":
                    status = "warning"
            
            # Distance from camera check (basic)
            nose_distance = abs(nose.z) if hasattr(nose, 'z') else 0
            if nose_distance > 0.1:
                feedback.append("💡 Try to sit closer to the camera for better detection")
            
            if not feedback:
                feedback.append("✅ Excellent posture - keep it up!")
            
            return PostureAnalysis(
                score=max(0, score),
                status=status,
                angles={
                    "neck": round(neck_angle, 2),
                    "back": round(back_angle, 2),
                    "knee": round(knee_angle, 2)
                },
                feedback=feedback,
                timestamp=datetime.now().isoformat(),
                keypoints={
                    "nose": {"x": nose.x, "y": nose.y},
                    "left_shoulder": {"x": left_shoulder.x, "y": left_shoulder.y},
                    "right_shoulder": {"x": right_shoulder.x, "y": right_shoulder.y},
                    "left_hip": {"x": left_hip.x, "y": left_hip.y},
                    "right_hip": {"x": right_hip.x, "y": right_hip.y},
                    "left_knee": {"x": left_knee.x, "y": left_knee.y},
                    "right_knee": {"x": right_knee.x, "y": right_knee.y}
                }
            )
        except Exception as e:
            logger.error(f"Error in posture analysis: {str(e)}")
            return PostureAnalysis(
                score=0,
                status="error",
                angles={"neck": 0, "back": 0, "knee": 0},
                feedback=[f"❌ Analysis error: {str(e)}"],
                timestamp=datetime.now().isoformat()
            )

def process_frame(frame_data: str) -> Optional[PostureAnalysis]:
    """Process a single frame and return posture analysis"""
    try:
        # Handle both formats: with and without data URL prefix
        if frame_data.startswith('data:image'):
            # Remove data URL prefix
            image_data = base64.b64decode(frame_data.split(',')[1])
        else:
            # Direct base64 data
            image_data = base64.b64decode(frame_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode image")
            return None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose_detector.process(rgb_frame)
        
        if results.pose_landmarks:
            # Analyze posture
            analysis = PostureRules.analyze_sitting_posture(results.pose_landmarks.landmark)
            logger.info(f"Posture analysis: {analysis.status} - Score: {analysis.score}")
            return analysis
        else:
            return PostureAnalysis(
                score=0,
                status="no_detection",
                angles={"neck": 0, "back": 0, "knee": 0},
                feedback=["❌ No person detected - make sure you're visible in the camera"],
                timestamp=datetime.now().isoformat()
            )
            
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return PostureAnalysis(
            score=0,
            status="error",
            angles={"neck": 0, "back": 0, "knee": 0},
            feedback=[f"❌ Processing error: {str(e)}"],
            timestamp=datetime.now().isoformat()
        )

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"Client connected. Total clients: {len(connected_clients)}")
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            message = json.loads(data)
            
            logger.info(f"Received message type: {message.get('type', 'unknown')}")
            
            # Process frame based on message type
            if message.get("type") == "analyze_frame":
                frame_data = message.get("frame")
                if frame_data:
                    analysis = process_frame(frame_data)
                    if analysis:
                        # Send analysis back to client with proper type
                        response = {
                            "type": "posture_analysis",
                            "analysis": analysis.dict()
                        }
                        await websocket.send_text(json.dumps(response))
                        logger.info(f"Sent analysis: {analysis.status}")
                    else:
                        logger.error("Failed to process frame")
                else:
                    logger.error("No frame data in message")
            else:
                logger.warning(f"Unknown message type: {message.get('type')}")
            
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(connected_clients)}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        connected_clients.discard(websocket)

# REST API endpoints
@app.get("/")
async def read_root():
    return {"message": "PostureGuard Backend API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connected_clients": len(connected_clients)
    }

@app.post("/analyze")
async def analyze_posture(frame_data: dict):
    """Analyze posture from a single frame"""
    try:
        analysis = process_frame(frame_data["frame"])
        if analysis:
            return analysis
        else:
            raise HTTPException(status_code=400, detail="Failed to process frame")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Static endpoint for serving a simple test page
@app.get("/test", response_class=HTMLResponse)
async def test_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PostureGuard Test</title>
    </head>
    <body>
        <h1>PostureGuard Backend Test</h1>
        <p>WebSocket endpoint: ws://localhost:8000/ws</p>
        <p>API endpoint: http://localhost:8000/analyze</p>
        <p>Health check: <a href="/health">http://localhost:8000/health</a></p>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )