PostureGuard 🎯
Advanced AI-Powered Posture Detection & Analysis System
Overview
PostureGuard is a real-time posture analysis application that uses computer vision and machine learning to monitor and analyze your sitting posture. The system provides instant feedback, tracks posture metrics, and helps improve your ergonomic health.
Features

Real-time Posture Analysis: Live webcam feed analysis with instant feedback
Video Upload Support: Analyze posture from uploaded video files
Rule-based Logic: Intelligent posture evaluation based on neck angle, back straightness, and shoulder alignment
Interactive Dashboard: Beautiful UI with real-time metrics and visualizations
Session Statistics: Track your posture quality over time
Pose Estimation: Visual skeleton overlay with joint detection
Demo Mode: Works offline with simulated data when backend is unavailable

Tech Stack
Frontend

React 18 - Modern JavaScript framework
Tailwind CSS - Utility-first CSS framework
Lucide React - Beautiful icon library
HTML5 Canvas - For pose visualization
WebRTC - Camera access and video streaming
WebSocket - Real-time communication with backend

Backend

FastAPI - Modern Python web framework
MediaPipe - Google's ML framework for pose detection
OpenCV - Computer vision library
NumPy - Numerical computing
Uvicorn - ASGI server
WebSocket - Real-time communication
Pydantic - Data validation and serialization

AI/ML

MediaPipe Pose - Human pose estimation
Rule-based Analysis - Custom posture evaluation logic
Computer Vision - Image processing and analysis


Deployment
Option 1: Local Development

Start Backend
bashcd backend
python main.py

Start Frontend (in new terminal)
bashcd frontend
npm start

Access Application

Frontend: http://localhost:3000
Backend API: http://localhost:8000
API Documentation: http://localhost:8000/docs