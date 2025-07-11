import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Upload, Play, Square, AlertTriangle, CheckCircle, Activity, BarChart3, Settings, Download } from 'lucide-react';

const PostureDetectionApp = () => {
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentPosture, setCurrentPosture] = useState({
    score: 95,
    status: 'good',
    angles: { neck: 15, back: 170, knee: 160 },
    feedback: ['System ready - Position yourself in front of the camera'],
    timestamp: new Date().toLocaleTimeString()
  });
  const [postureHistory, setPostureHistory] = useState([]);
  const [videoFile, setVideoFile] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [sessionStats, setSessionStats] = useState({
    totalFrames: 0,
    goodPostures: 0,
    warnings: 0,
    badPostures: 0
  });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);
  const analysisIntervalRef = useRef(null);

  // WebSocket connection for real-time communication with backend
  const connectWebSocket = useCallback(() => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8000/ws');
      
      wsRef.current.onopen = () => {
        setIsConnected(true);
        console.log('Connected to backend WebSocket');
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'posture_analysis') {
          updatePostureAnalysis(data.analysis);
        }
      };

      wsRef.current.onclose = () => {
        setIsConnected(false);
        console.log('Disconnected from backend');
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };
    } catch (error) {
      console.error('Failed to connect to backend:', error);
      // Fallback to demo mode
      startDemoMode();
    }
  }, []);

  // Initialize WebSocket connection
  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  // Start webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsWebcamActive(true);
      }
    } catch (error) {
      console.error('Error accessing webcam:', error);
      alert('Error accessing webcam: ' + error.message);
    }
  };

  // Stop webcam
  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);
    if (isAnalyzing) {
      stopAnalysis();
    }
  };

  // Handle video file upload
  const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      if (videoRef.current) {
        videoRef.current.src = url;
        videoRef.current.load();
      }
      setVideoFile(file);
    }
  };

  // Start posture analysis
  const startAnalysis = () => {
    if (!isWebcamActive && !videoFile) {
      alert('Please start webcam or upload a video first');
      return;
    }

    setIsAnalyzing(true);
    
    // Start sending frames to backend for analysis
    if (isConnected && wsRef.current) {
      analysisIntervalRef.current = setInterval(() => {
        captureAndSendFrame();
      }, 200); // Send frame every 200ms
    } else {
      // Fallback to demo mode
      startDemoMode();
    }
  };

  // Stop posture analysis
  const stopAnalysis = () => {
    setIsAnalyzing(false);
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);
      analysisIntervalRef.current = null;
    }
  };

  // Capture frame and send to backend
  const captureAndSendFrame = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;

    // Set canvas size
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to base64 and send to backend
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'analyze_frame',
        image: imageData,
        timestamp: Date.now()
      }));
    }
  };

  // Demo mode for when backend is not available
  const startDemoMode = () => {
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);
    }
    
    analysisIntervalRef.current = setInterval(() => {
      const simulatedAnalysis = generateSimulatedAnalysis();
      updatePostureAnalysis(simulatedAnalysis);
    }, 1000);
  };

  // Generate simulated posture analysis
  const generateSimulatedAnalysis = () => {
    const time = Date.now() * 0.001;
    const variation = Math.sin(time * 0.5) * 20 + Math.random() * 10;
    
    const neckAngle = Math.max(0, Math.min(45, 15 + variation));
    const backAngle = Math.max(140, Math.min(180, 170 + variation * 0.5));
    const kneeAngle = Math.max(90, Math.min(180, 160 + variation * 0.8));

    const feedback = [];
    let score = 100;
    let status = 'good';

    // Rule-based analysis
    if (neckAngle > 30) {
      feedback.push('⚠️ Neck is too forward - straighten your neck');
      score -= 15;
      status = 'warning';
    }

    if (backAngle < 150) {
      feedback.push('⚠️ Back is hunched - sit up straight');
      score -= 20;
      status = 'bad';
    }

    if (kneeAngle < 90) {
      feedback.push('⚠️ Knee position may cause strain');
      score -= 10;
      if (status === 'good') status = 'warning';
    }

    if (feedback.length === 0) {
      feedback.push('✅ Excellent posture - keep it up!');
    }

    return {
      score: Math.max(0, score),
      status,
      angles: { neck: neckAngle, back: backAngle, knee: kneeAngle },
      feedback,
      timestamp: new Date().toLocaleTimeString(),
      keypoints: generateKeypoints()
    };
  };

  // Generate simulated keypoints
  const generateKeypoints = () => {
    return {
      nose: { x: 320, y: 100, confidence: 0.9 },
      neck: { x: 320, y: 150, confidence: 0.9 },
      shoulders: { x: 320, y: 200, confidence: 0.9 },
      spine: { x: 320, y: 300, confidence: 0.9 },
      hips: { x: 320, y: 400, confidence: 0.9 },
      knees: { x: 320, y: 450, confidence: 0.9 },
      ankles: { x: 320, y: 500, confidence: 0.9 }
    };
  };

  // Update posture analysis
  const updatePostureAnalysis = (analysis) => {
    setCurrentPosture(analysis);
    
    // Update history
    setPostureHistory(prev => {
      const newHistory = [...prev, analysis];
      if (newHistory.length > 100) {
        newHistory.shift();
      }
      return newHistory;
    });

    // Update session stats
    setSessionStats(prev => ({
      totalFrames: prev.totalFrames + 1,
      goodPostures: prev.goodPostures + (analysis.status === 'good' ? 1 : 0),
      warnings: prev.warnings + (analysis.status === 'warning' ? 1 : 0),
      badPostures: prev.badPostures + (analysis.status === 'bad' ? 1 : 0)
    }));

    // Draw pose estimation if keypoints are available
    if (analysis.keypoints && canvasRef.current) {
      drawPoseEstimation(analysis.keypoints, analysis.angles);
    }
  };

  // Draw pose estimation on canvas
  const drawPoseEstimation = (keypoints, angles) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw keypoints
    ctx.fillStyle = '#3b82f6';
    Object.values(keypoints).forEach(point => {
      if (point.confidence > 0.5) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });

    // Draw skeleton connections
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    const connections = [
      ['nose', 'neck'],
      ['neck', 'shoulders'],
      ['shoulders', 'spine'],
      ['spine', 'hips'],
      ['hips', 'knees'],
      ['knees', 'ankles']
    ];

    connections.forEach(([from, to]) => {
      const fromPoint = keypoints[from];
      const toPoint = keypoints[to];
      
      if (fromPoint?.confidence > 0.5 && toPoint?.confidence > 0.5) {
        ctx.beginPath();
        ctx.moveTo(fromPoint.x, fromPoint.y);
        ctx.lineTo(toPoint.x, toPoint.y);
        ctx.stroke();
      }
    });

    // Draw angle indicators
    ctx.font = '14px Arial';
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1;

    const angleText = `Neck: ${Math.round(angles.neck)}° | Back: ${Math.round(angles.back)}° | Knee: ${Math.round(angles.knee)}°`;
    ctx.strokeText(angleText, 10, 25);
    ctx.fillText(angleText, 10, 25);
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'good': return 'text-green-600 bg-green-100';
      case 'warning': return 'text-yellow-600 bg-yellow-100';
      case 'bad': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  // Get status icon
  const getStatusIcon = (status) => {
    switch (status) {
      case 'good': return <CheckCircle className="w-5 h-5" />;
      case 'warning': return <AlertTriangle className="w-5 h-5" />;
      case 'bad': return <AlertTriangle className="w-5 h-5" />;
      default: return <Activity className="w-5 h-5" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-100 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 bg-white rounded-2xl shadow-lg p-6">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🎯 PostureGuard
          </h1>
          <p className="text-gray-600 text-lg">
            Advanced AI-Powered Posture Detection & Analysis System
          </p>
          <div className="flex items-center justify-center mt-4 space-x-4">
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${isConnected ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm font-medium">
                {isConnected ? 'Backend Connected' : 'Demo Mode'}
              </span>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Video Section */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
              <Camera className="w-6 h-6 mr-2" />
              Video Input & Analysis
            </h2>

            {/* Video Container */}
            <div className="relative bg-black rounded-xl overflow-hidden mb-4" style={{ height: '400px' }}>
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                autoPlay
                muted
                playsInline
              />
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full pointer-events-none"
              />
              {!isWebcamActive && !videoFile && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-white text-center">
                    <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p className="text-lg">Start webcam or upload video</p>
                  </div>
                </div>
              )}
            </div>

            {/* Controls */}
            <div className="flex flex-wrap gap-3 mb-4">
              <button
                onClick={startWebcam}
                disabled={isWebcamActive}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Camera className="w-4 h-4" />
                <span>Start Webcam</span>
              </button>

              <button
                onClick={stopWebcam}
                disabled={!isWebcamActive}
                className="flex items-center space-x-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Square className="w-4 h-4" />
                <span>Stop Webcam</span>
              </button>

              <label className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 cursor-pointer transition-colors">
                <Upload className="w-4 h-4" />
                <span>Upload Video</span>
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleVideoUpload}
                  className="hidden"
                />
              </label>

              <button
                onClick={isAnalyzing ? stopAnalysis : startAnalysis}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-white transition-colors ${
                  isAnalyzing 
                    ? 'bg-red-600 hover:bg-red-700' 
                    : 'bg-purple-600 hover:bg-purple-700'
                }`}
              >
                {isAnalyzing ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                <span>{isAnalyzing ? 'Stop Analysis' : 'Start Analysis'}</span>
              </button>
            </div>

            {/* Status */}
            <div className={`p-4 rounded-lg ${getStatusColor(currentPosture.status)}`}>
              <div className="flex items-center space-x-2">
                {getStatusIcon(currentPosture.status)}
                <span className="font-semibold">
                  Status: {currentPosture.status.charAt(0).toUpperCase() + currentPosture.status.slice(1)}
                </span>
              </div>
              <p className="mt-1">Score: {currentPosture.score}/100</p>
            </div>

            {/* Instructions */}
            <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h3 className="font-semibold text-blue-800 mb-2">Instructions:</h3>
              <p className="text-blue-700 text-sm">
                Sit normally or perform squats in front of the camera. The system will analyze your posture in real-time and provide feedback based on rule-based logic for neck angle, back straightness, and knee position.
              </p>
            </div>
          </div>

          {/* Analysis Section */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
              <BarChart3 className="w-6 h-6 mr-2" />
              Posture Analysis
            </h2>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-4 rounded-xl">
                <div className="text-2xl font-bold">{currentPosture.score}</div>
                <div className="text-sm opacity-90">Posture Score</div>
                <div className="w-full bg-white bg-opacity-20 rounded-full h-2 mt-2">
                  <div 
                    className="bg-white h-2 rounded-full transition-all duration-300"
                    style={{ width: `${currentPosture.score}%` }}
                  ></div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-green-500 to-teal-600 text-white p-4 rounded-xl">
                <div className="text-2xl font-bold">{Math.round(currentPosture.angles.neck)}°</div>
                <div className="text-sm opacity-90">Neck Angle</div>
                <div className="text-xs mt-1">
                  {currentPosture.angles.neck > 30 ? 'Too forward' : 'Good'}
                </div>
              </div>

              <div className="bg-gradient-to-r from-yellow-500 to-orange-600 text-white p-4 rounded-xl">
                <div className="text-2xl font-bold">{Math.round(currentPosture.angles.back)}°</div>
                <div className="text-sm opacity-90">Back Angle</div>
                <div className="text-xs mt-1">
                  {currentPosture.angles.back < 150 ? 'Hunched' : 'Straight'}
                </div>
              </div>

              <div className="bg-gradient-to-r from-red-500 to-pink-600 text-white p-4 rounded-xl">
                <div className="text-2xl font-bold">{Math.round(currentPosture.angles.knee)}°</div>
                <div className="text-sm opacity-90">Knee Angle</div>
                <div className="text-xs mt-1">
                  {currentPosture.angles.knee < 90 ? 'Strain risk' : 'Safe'}
                </div>
              </div>
            </div>

            {/* Real-time Feedback */}
            <div className="bg-gray-50 rounded-xl p-4 mb-6">
              <h3 className="font-semibold text-gray-800 mb-3 flex items-center">
                <Activity className="w-5 h-5 mr-2" />
                Real-time Feedback
              </h3>
              <div className="space-y-2">
                {currentPosture.feedback.map((item, index) => (
                  <div key={index} className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                    <div className="flex-1">
                      <p className="text-sm text-gray-700">{item}</p>
                      <p className="text-xs text-gray-500">{currentPosture.timestamp}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Rule-based Logic Display */}
            <div className="bg-blue-50 rounded-xl p-4">
              <h3 className="font-semibold text-blue-800 mb-3 flex items-center">
                <Settings className="w-5 h-5 mr-2" />
                Rule-based Logic
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Neck Angle:</span>
                  <span className="font-mono">{'> 30° = Warning'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Back Angle:</span>
                  <span className="font-mono">{'< 150° = Bad'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Knee Angle:</span>
                  <span className="font-mono">{'< 90° = Warning'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Session Statistics */}
        <div className="bg-white rounded-2xl shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
            <BarChart3 className="w-6 h-6 mr-2" />
            Session Statistics
          </h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center p-4 bg-blue-50 rounded-xl">
              <div className="text-2xl font-bold text-blue-600">{sessionStats.totalFrames}</div>
              <div className="text-sm text-gray-600">Total Frames</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-xl">
              <div className="text-2xl font-bold text-green-600">{sessionStats.goodPostures}</div>
              <div className="text-sm text-gray-600">Good Postures</div>
            </div>
            <div className="text-center p-4 bg-yellow-50 rounded-xl">
              <div className="text-2xl font-bold text-yellow-600">{sessionStats.warnings}</div>
              <div className="text-sm text-gray-600">Warnings</div>
            </div>
            <div className="text-center p-4 bg-red-50 rounded-xl">
              <div className="text-2xl font-bold text-red-600">{sessionStats.badPostures}</div>
              <div className="text-sm text-gray-600">Bad Postures</div>
            </div>
          </div>

          {sessionStats.totalFrames > 0 && (
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Posture Quality</span>
                  <span>{((sessionStats.goodPostures / sessionStats.totalFrames) * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${(sessionStats.goodPostures / sessionStats.totalFrames) * 100}%` }}
                  ></div>
                </div>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">
                  Average Score: {postureHistory.length > 0 ? 
                    Math.round(postureHistory.reduce((sum, entry) => sum + entry.score, 0) / postureHistory.length) : 0
                  }/100
                </span>
                <button className="flex items-center space-x-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors">
                  <Download className="w-4 h-4" />
                  <span>Export Data</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PostureDetectionApp;