# Smart Alarm SNN Web App Demo

This web application demonstrates the Smart Alarm system using Spiking Neural Networks (SNNs) from the Jupyter notebook.

## Features

🧠 **Spiking Neural Network**: Real-time sleep stage classification using brain-inspired computing  
📊 **Live Visualization**: Interactive charts showing sleep stages, heart rate, and movement  
⏰ **Smart Alarm**: Optimal wake-up timing during Light or REM sleep phases  
🔬 **Sleep Analysis**: Detailed sleep pattern analysis with comprehensive visualizations  

## How to Run

1. **Install Dependencies:**
   ```bash
   py -m pip install -r requirements.txt
   ```

2. **Start the Web App:**
   ```bash
   py app.py
   ```

3. **Open in Browser:**
   Navigate to `http://localhost:5000`

## Using the Demo

### 🎛️ Control Panel
- **Set Target Wake Time**: Choose your desired wake time (in hours after sleep starts)
- **Start Demo**: Begin the 8-hour sleep simulation
- **Stop Demo**: Halt the simulation at any time

### 📊 Real-Time Features
- **Live Status**: Current time, sleep stage, heart rate, movement data
- **Interactive Charts**: 
  - Sleep stages over time (stepped line chart)
  - Heart rate monitoring (smooth curve)
  - Movement activity (area chart)
  - Sleep statistics (real-time calculations)

### 🔬 Analysis Features
- **Generate Analysis**: Creates comprehensive sleep pattern visualizations
- **Sleep Metrics**: Total sleep time, stage distribution, physiological averages
- **SNN Performance**: Shows how well the neural network classifies sleep stages

## How the Smart Alarm Works

1. **Sleep Monitoring**: SNN continuously processes sensor data (accelerometer + heart rate)
2. **Stage Classification**: Predicts current sleep stage (Awake, Light, Deep, REM)
3. **Alarm Window**: Looks for optimal wake opportunities 30 minutes before target time
4. **Optimal Triggering**: Wakes you during Light or REM sleep when possible
5. **Fallback**: Forces alarm at target time if no optimal window found

## Technical Details

### Architecture
- **Backend**: Flask web server with real-time data processing
- **Frontend**: HTML5/CSS3/JavaScript with Chart.js for visualizations
- **SNN Model**: Custom Leaky Integrate-and-Fire neurons with STDP learning
- **Data Processing**: Multi-modal spike encoding (threshold, rate, delta-based)

### Key Components
- `snn_models.py`: Core SNN classes and sleep data generation
- `app.py`: Flask application with REST API endpoints
- `templates/index.html`: Web interface with responsive design
- `static/app.js`: Frontend JavaScript for real-time interactions

## Sleep Stages Explained

- **🟥 Awake**: High activity, elevated heart rate, frequent movement
- **🟦 Light Sleep**: Easy to wake, moderate activity, optimal for alarms
- **🟫 Deep Sleep**: Hard to wake, minimal movement, physical restoration
- **🟨 REM Sleep**: Dream stage, paralyzed muscles, optimal for alarms

## Demo Highlights

- **Realistic Sleep Data**: 8-hour synthetic sleep patterns with natural progressions
- **Real-Time Processing**: 100ms updates simulating actual wearable device operation
- **Visual Feedback**: Color-coded sleep stages and alarm status indicators
- **Performance Metrics**: Live accuracy tracking and system diagnostics

## Browser Compatibility

- ✅ Chrome 70+
- ✅ Firefox 65+
- ✅ Safari 12+
- ✅ Edge 79+

## Performance Notes

- The demo runs at accelerated speed (100ms per minute of sleep data)
- Real implementation would process data at 1-minute intervals
- Charts automatically limit to last 100 data points for smooth performance
- Background processing uses threading for non-blocking operation

Enjoy exploring the future of sleep technology! 🌙✨