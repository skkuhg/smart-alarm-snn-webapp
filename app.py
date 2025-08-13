"""
Smart Alarm SNN Web App
Flask backend with real-time demo functionality
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import threading
import time
from datetime import datetime
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Import our SNN models
from snn_models import (
    SleepDataGenerator, 
    DataPreprocessor, 
    SpikingNeuralNetwork, 
    SmartAlarmSystem
)

app = Flask(__name__)
CORS(app)

# Global variables for the demo
demo_running = False
demo_thread = None
demo_data = []
current_time = 0
alarm_system = None
sleep_data = None
snn = None
preprocessor = None

# Initialize the system
def initialize_system():
    global snn, preprocessor, alarm_system, sleep_data
    
    # Generate synthetic sleep data
    generator = SleepDataGenerator(duration_hours=8, sampling_rate=1)
    sleep_data = generator.generate_dataset()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(spike_threshold_percentile=70)
    spike_data = preprocessor.convert_to_spikes(sleep_data)
    
    # Initialize SNN
    snn = SpikingNeuralNetwork(input_size=4, hidden_size=16, output_size=4)
    
    # Quick training (simplified for demo)
    X = spike_data
    y = spike_data['sleep_stage']
    
    # Simple training with a few epochs
    for epoch in range(2):
        for idx in range(min(100, len(X))):  # Train on subset for demo
            input_features = [
                X.iloc[idx]['accelerometer_x_spike_threshold'],
                X.iloc[idx]['accelerometer_y_spike_threshold'],
                X.iloc[idx]['accelerometer_z_spike_threshold'],
                X.iloc[idx]['heart_rate_spike_threshold']
            ]
            snn.forward(input_features, idx)
        snn.reset_network()
    
    # Initialize alarm system
    alarm_system = SmartAlarmSystem(snn, preprocessor, alarm_window_minutes=30)
    
    print("System initialized successfully!")

# Initialize on startup
initialize_system()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_demo', methods=['POST'])
def start_demo():
    global demo_running, demo_thread, current_time, demo_data
    
    if demo_running:
        return jsonify({'status': 'error', 'message': 'Demo already running'})
    
    data = request.get_json()
    target_wake_time = data.get('target_wake_time', 420)  # Default 7 hours
    
    # Reset variables
    current_time = 0
    demo_data = []
    alarm_system.sleep_history = []
    alarm_system.alarm_triggered = False
    alarm_system.set_alarm(target_wake_time)
    
    # Start demo thread
    demo_running = True
    demo_thread = threading.Thread(target=run_demo)
    demo_thread.start()
    
    return jsonify({'status': 'success', 'message': 'Demo started'})

@app.route('/api/stop_demo', methods=['POST'])
def stop_demo():
    global demo_running
    demo_running = False
    return jsonify({'status': 'success', 'message': 'Demo stopped'})

@app.route('/api/demo_status')
def demo_status():
    global demo_running, current_time, demo_data
    
    # Get recent data points (last 10)
    recent_data = demo_data[-10:] if len(demo_data) > 10 else demo_data
    
    return jsonify({
        'running': demo_running,
        'current_time': current_time,
        'data_points': len(demo_data),
        'recent_data': recent_data,
        'alarm_triggered': alarm_system.alarm_triggered if alarm_system else False
    })

@app.route('/api/full_data')
def get_full_data():
    global demo_data
    return jsonify({
        'data': demo_data,
        'sleep_stages': ['Awake', 'Light', 'Deep', 'REM'],
        'stage_colors': {
            'Awake': '#ff6b6b',
            'Light': '#4ecdc4', 
            'Deep': '#45b7d1',
            'REM': '#f9ca24'
        }
    })

@app.route('/api/sleep_analysis')
def sleep_analysis():
    global sleep_data
    
    try:
        if sleep_data is None:
            return jsonify({'error': 'No sleep data available'})
        
        # Create a copy to avoid modifying global data
        data_copy = sleep_data.copy()
        
        # Generate analysis plot
        plt.style.use('default')  # Reset style
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        # Sleep stages over time
        ax1 = axes[0, 0]
        stage_mapping = {'Awake': 3, 'REM': 2, 'Light': 1, 'Deep': 0}
        data_copy['stage_numeric'] = data_copy['sleep_stage'].map(stage_mapping)
        time_hours = data_copy['timestamp'] / 60
        
        ax1.plot(time_hours, data_copy['stage_numeric'], 'b-', linewidth=2)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Sleep Stage')
        ax1.set_title('Sleep Stages Over Time')
        ax1.set_yticks([0, 1, 2, 3])
        ax1.set_yticklabels(['Deep', 'Light', 'REM', 'Awake'])
        ax1.grid(True, alpha=0.3)
        
        # Heart rate over time
        ax2 = axes[0, 1]
        ax2.plot(time_hours, data_copy['heart_rate'], 'r-', linewidth=1)
        ax2.fill_between(time_hours, data_copy['heart_rate'], alpha=0.3, color='red')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Heart Rate (BPM)')
        ax2.set_title('Heart Rate During Sleep')
        ax2.grid(True, alpha=0.3)
        
        # Movement over time
        ax3 = axes[1, 0]
        ax3.plot(time_hours, data_copy['accel_magnitude'], 'purple', linewidth=1)
        ax3.fill_between(time_hours, data_copy['accel_magnitude'], alpha=0.3, color='purple')
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Movement (g)')
        ax3.set_title('Movement During Sleep')
        ax3.grid(True, alpha=0.3)
        
        # Sleep stage distribution
        ax4 = axes[1, 1]
        stage_counts = data_copy['sleep_stage'].value_counts()
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        wedges, texts, autotexts = ax4.pie(stage_counts.values, labels=stage_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax4.set_title('Sleep Stage Distribution')
        
        plt.tight_layout()
        
        # Convert plot to base64 image
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close(fig)  # Close the specific figure
        
        graphic = base64.b64encode(image_png).decode('utf-8')
        
        return jsonify({
            'plot': graphic,
            'statistics': {
                'total_sleep_time': int(len(data_copy[data_copy['sleep_stage'] != 'Awake'])),
                'stage_distribution': data_copy['sleep_stage'].value_counts().to_dict(),
                'avg_heart_rate': float(data_copy['heart_rate'].mean()),
                'avg_movement': float(data_copy['accel_magnitude'].mean())
            }
        })
        
    except Exception as e:
        print(f"Error generating sleep analysis: {e}")
        return jsonify({
            'error': f'Failed to generate analysis: {str(e)}',
            'plot': None,
            'statistics': {
                'total_sleep_time': 0,
                'stage_distribution': {},
                'avg_heart_rate': 0,
                'avg_movement': 0
            }
        })

def run_demo():
    """Run the sleep monitoring demo"""
    global demo_running, current_time, demo_data, sleep_data
    
    print("Demo started!")
    
    for idx, row in sleep_data.iterrows():
        if not demo_running:
            break
            
        current_time = row['timestamp']
        
        # Simulate sensor data
        sensor_data = {
            'accel_x': row['accelerometer_x'],
            'accel_y': row['accelerometer_y'],
            'accel_z': row['accelerometer_z'],
            'heart_rate': row['heart_rate']
        }
        
        # Process through alarm system
        predicted_stage, alarm_decision = alarm_system.process_real_time_data(
            sensor_data, current_time
        )
        
        # Store data point
        data_point = {
            'time': current_time,
            'time_hours': current_time / 60,
            'true_stage': row['sleep_stage'],
            'predicted_stage': predicted_stage,
            'heart_rate': row['heart_rate'],
            'movement': row['accel_magnitude'],
            'alarm_status': alarm_decision['reason'],
            'alarm_triggered': alarm_decision['trigger']
        }
        
        demo_data.append(data_point)
        
        # Check if alarm triggered
        if alarm_decision['trigger']:
            print(f"ALARM TRIGGERED at {current_time} minutes!")
            break
        
        # Sleep for demo timing (slower for stability)
        time.sleep(0.5)  # 500ms per minute of sleep data
    
    demo_running = False
    print("Demo completed!")

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')