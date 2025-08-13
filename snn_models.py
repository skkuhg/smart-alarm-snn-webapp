"""
Smart Alarm SNN Models - Extracted from Jupyter Notebook
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import time
from datetime import datetime, timedelta
import random

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)


class SpikingNeuron:
    def __init__(self, threshold=1.0, decay=0.9, refractory_period=2):
        self.threshold = threshold
        self.decay = decay
        self.refractory_period = refractory_period
        self.potential = 0.0
        self.last_spike_time = -float('inf')
        self.spike_times = []
        
    def update(self, input_current, time_step):
        """Update neuron state and check for spikes"""
        # Check if in refractory period
        if time_step - self.last_spike_time <= self.refractory_period:
            return False
        
        # Update membrane potential
        self.potential = self.potential * self.decay + input_current
        
        # Check for spike
        if self.potential >= self.threshold:
            self.potential = 0.0  # Reset after spike
            self.last_spike_time = time_step
            self.spike_times.append(time_step)
            return True
        
        return False
    
    def reset(self):
        """Reset neuron state"""
        self.potential = 0.0
        self.last_spike_time = -float('inf')
        self.spike_times = []


class STDPSynapse:
    def __init__(self, weight=0.5, learning_rate=0.01, tau_plus=20, tau_minus=20):
        self.weight = weight
        self.learning_rate = learning_rate
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = 0.1
        self.A_minus = -0.12
    
    def stdp_update(self, pre_spike_time, post_spike_time):
        """Update weight using STDP rule"""
        if pre_spike_time is None or post_spike_time is None:
            return
        
        dt = post_spike_time - pre_spike_time
        
        if dt > 0:  # Pre before post - potentiation
            dw = self.A_plus * np.exp(-dt / self.tau_plus)
        elif dt < 0:  # Post before pre - depression
            dw = self.A_minus * np.exp(dt / self.tau_minus)
        else:
            dw = 0
        
        self.weight += self.learning_rate * dw
        self.weight = np.clip(self.weight, 0.0, 2.0)  # Keep weights in bounds


class SpikingNeuralNetwork:
    def __init__(self, input_size=4, hidden_size=20, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Create neurons
        self.input_neurons = [SpikingNeuron() for _ in range(input_size)]
        self.hidden_neurons = [SpikingNeuron() for _ in range(hidden_size)]
        self.output_neurons = [SpikingNeuron() for _ in range(output_size)]
        
        # Create synapses
        self.input_hidden_synapses = [[STDPSynapse(np.random.uniform(0.1, 0.9)) 
                                      for _ in range(hidden_size)] 
                                     for _ in range(input_size)]
        
        self.hidden_output_synapses = [[STDPSynapse(np.random.uniform(0.1, 0.9)) 
                                       for _ in range(output_size)] 
                                      for _ in range(hidden_size)]
        
        self.sleep_stage_map = {'Awake': 0, 'Light': 1, 'Deep': 2, 'REM': 3}
        self.reverse_stage_map = {v: k for k, v in self.sleep_stage_map.items()}
    
    def forward(self, input_spikes, time_step):
        """Forward pass through the network"""
        # Update input neurons
        input_spike_states = []
        for i, neuron in enumerate(self.input_neurons):
            spike_occurred = neuron.update(input_spikes[i], time_step)
            input_spike_states.append(spike_occurred)
        
        # Update hidden neurons
        hidden_spike_states = []
        for j, neuron in enumerate(self.hidden_neurons):
            total_input = 0
            for i in range(self.input_size):
                if input_spike_states[i]:
                    total_input += self.input_hidden_synapses[i][j].weight
            
            spike_occurred = neuron.update(total_input, time_step)
            hidden_spike_states.append(spike_occurred)
        
        # Update output neurons
        output_spike_states = []
        for k, neuron in enumerate(self.output_neurons):
            total_input = 0
            for j in range(self.hidden_size):
                if hidden_spike_states[j]:
                    total_input += self.hidden_output_synapses[j][k].weight
            
            spike_occurred = neuron.update(total_input, time_step)
            output_spike_states.append(spike_occurred)
        
        return output_spike_states
    
    def predict(self, input_data):
        """Make predictions on new data"""
        predictions = []
        
        # Reset network
        self.reset_network()
        
        for idx in range(len(input_data)):
            input_features = [
                input_data.iloc[idx]['accelerometer_x_spike_threshold'],
                input_data.iloc[idx]['accelerometer_y_spike_threshold'],
                input_data.iloc[idx]['accelerometer_z_spike_threshold'],
                input_data.iloc[idx]['heart_rate_spike_threshold']
            ]
            
            # Forward pass
            self.forward(input_features, idx)
            
            # Get prediction based on accumulated spikes
            output_activities = [len(neuron.spike_times) for neuron in self.output_neurons]
            predicted_class = np.argmax(output_activities) if max(output_activities) > 0 else 1  # Default to Light
            
            predictions.append(self.reverse_stage_map[predicted_class])
        
        return predictions
    
    def reset_network(self):
        """Reset all neurons in the network"""
        for neuron in self.input_neurons + self.hidden_neurons + self.output_neurons:
            neuron.reset()


class DataPreprocessor:
    def __init__(self, spike_threshold_percentile=75):
        self.scaler = StandardScaler()
        self.spike_threshold_percentile = spike_threshold_percentile
        self.spike_thresholds = {}
        
    def normalize_data(self, data):
        """Normalize sensor data"""
        sensor_columns = ['accelerometer_x', 'accelerometer_y', 'accelerometer_z', 
                         'heart_rate', 'accel_magnitude']
        
        data_normalized = data.copy()
        data_normalized[sensor_columns] = self.scaler.fit_transform(data[sensor_columns])
        
        return data_normalized
    
    def convert_to_spikes(self, data, window_size=5):
        """Convert sensor data to spike trains"""
        data_normalized = self.normalize_data(data)
        
        # Calculate spike thresholds for each sensor
        sensor_columns = ['accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'heart_rate']
        
        for col in sensor_columns:
            self.spike_thresholds[col] = np.percentile(
                np.abs(data_normalized[col]), self.spike_threshold_percentile
            )
        
        # Generate spike trains
        spike_data = data_normalized.copy()
        
        for col in sensor_columns:
            # Method 1: Threshold-based spikes
            spike_data[f'{col}_spike_threshold'] = (
                np.abs(data_normalized[col]) > self.spike_thresholds[col]
            ).astype(int)
            
            # Method 2: Rate-based encoding (spike frequency proportional to signal strength)
            col_min = data_normalized[col].min()
            col_max = data_normalized[col].max()
            
            # Handle case where min equals max (avoid division by zero)
            if col_max - col_min == 0:
                normalized_signal = pd.Series([0.5] * len(data_normalized[col]), index=data_normalized[col].index)
            else:
                normalized_signal = (data_normalized[col] - col_min) / (col_max - col_min)
            
            # Ensure values are finite and within [0, 1] range
            normalized_signal = normalized_signal.fillna(0.5)
            normalized_signal = np.clip(normalized_signal, 0, 1)
            
            spike_data[f'{col}_spike_rate'] = (normalized_signal * 10).astype(int)
            
            # Method 3: Delta-based spikes (spikes on significant changes)
            delta = np.abs(np.diff(data_normalized[col], prepend=data_normalized[col].iloc[0]))
            delta_threshold = np.percentile(delta, self.spike_threshold_percentile)
            spike_data[f'{col}_spike_delta'] = (delta > delta_threshold).astype(int)
        
        # Aggregate spike features
        spike_data['total_spikes'] = spike_data[[col for col in spike_data.columns if 'spike_threshold' in col]].sum(axis=1)
        spike_data['accel_spike_sum'] = spike_data[['accelerometer_x_spike_threshold', 
                                                   'accelerometer_y_spike_threshold', 
                                                   'accelerometer_z_spike_threshold']].sum(axis=1)
        
        return spike_data


class SleepDataGenerator:
    def __init__(self, duration_hours=8, sampling_rate=1):
        self.duration_hours = duration_hours
        self.sampling_rate = sampling_rate
        self.total_samples = duration_hours * 60 * sampling_rate
        
        # Sleep stage parameters
        self.stage_params = {
            'Awake': {
                'accel_mean': 0.3, 'accel_std': 0.2,
                'hr_mean': 75, 'hr_std': 8
            },
            'Light': {
                'accel_mean': 0.1, 'accel_std': 0.05,
                'hr_mean': 65, 'hr_std': 5
            },
            'Deep': {
                'accel_mean': 0.02, 'accel_std': 0.01,
                'hr_mean': 58, 'hr_std': 3
            },
            'REM': {
                'accel_mean': 0.05, 'accel_std': 0.03,
                'hr_mean': 68, 'hr_std': 6
            }
        }
    
    def generate_sleep_cycle(self):
        """Generate realistic sleep stage progression"""
        cycle_length = 90  # minutes
        num_cycles = int(self.duration_hours * 60 / cycle_length)
        
        sleep_stages = []
        
        # Initial awake period (5-15 minutes)
        initial_awake = np.random.randint(5, 16)
        sleep_stages.extend(['Awake'] * initial_awake)
        
        for cycle in range(num_cycles):
            cycle_stages = [
                ('Light', np.random.randint(10, 20)),
                ('Deep', np.random.randint(20, 40)),
                ('Light', np.random.randint(5, 15)),
                ('REM', np.random.randint(10, 25))
            ]
            
            # Add some variability for later cycles (more REM, less deep sleep)
            if cycle > num_cycles // 2:
                cycle_stages[1] = ('Deep', np.random.randint(5, 15))
                cycle_stages[3] = ('REM', np.random.randint(15, 30))
            
            for stage, duration in cycle_stages:
                sleep_stages.extend([stage] * duration)
        
        # Final awakening
        final_awake = np.random.randint(5, 15)
        sleep_stages.extend(['Awake'] * final_awake)
        
        # Trim or extend to match total samples
        if len(sleep_stages) > self.total_samples:
            sleep_stages = sleep_stages[:self.total_samples]
        elif len(sleep_stages) < self.total_samples:
            remaining = self.total_samples - len(sleep_stages)
            sleep_stages.extend(['Light'] * remaining)
        
        return sleep_stages
    
    def generate_sensor_data(self, sleep_stages):
        """Generate accelerometer and heart rate data based on sleep stages"""
        data = []
        
        for i, stage in enumerate(sleep_stages):
            params = self.stage_params[stage]
            
            # Generate accelerometer data (3-axis)
            accel_magnitude = np.random.normal(params['accel_mean'], params['accel_std'])
            accel_magnitude = max(0, accel_magnitude)
            
            # Distribute magnitude across 3 axes
            accel_x = np.random.normal(0, accel_magnitude/3)
            accel_y = np.random.normal(0, accel_magnitude/3)
            accel_z = np.random.normal(0, accel_magnitude/3)
            
            # Generate heart rate data
            heart_rate = np.random.normal(params['hr_mean'], params['hr_std'])
            heart_rate = max(40, min(120, heart_rate))
            
            # Add some temporal correlation (smooth transitions)
            if i > 0:
                prev_hr = data[-1]['heart_rate']
                heart_rate = 0.8 * prev_hr + 0.2 * heart_rate
            
            data.append({
                'timestamp': i,
                'accelerometer_x': accel_x,
                'accelerometer_y': accel_y,
                'accelerometer_z': accel_z,
                'heart_rate': heart_rate,
                'sleep_stage': stage
            })
        
        return pd.DataFrame(data)
    
    def generate_dataset(self):
        """Generate complete sleep dataset"""
        sleep_stages = self.generate_sleep_cycle()
        dataset = self.generate_sensor_data(sleep_stages)
        
        # Add derived features
        dataset['accel_magnitude'] = np.sqrt(
            dataset['accelerometer_x']**2 + 
            dataset['accelerometer_y']**2 + 
            dataset['accelerometer_z']**2
        )
        
        # Rolling averages for smoothing
        dataset['hr_rolling_mean'] = dataset['heart_rate'].rolling(window=5, center=True).mean()
        dataset['accel_rolling_mean'] = dataset['accel_magnitude'].rolling(window=5, center=True).mean()
        
        return dataset


class SmartAlarmSystem:
    def __init__(self, snn_model, preprocessor, alarm_window_minutes=30):
        self.snn_model = snn_model
        self.preprocessor = preprocessor
        self.alarm_window_minutes = alarm_window_minutes
        self.alarm_triggered = False
        self.target_wake_time = None
        self.optimal_stages = ['Light', 'REM']
        self.sleep_history = []
        
    def set_alarm(self, target_wake_time_minutes):
        """Set target wake time (in minutes from sleep start)"""
        self.target_wake_time = target_wake_time_minutes
        self.alarm_triggered = False
        
    def process_real_time_data(self, sensor_data, current_time_minutes):
        """Process incoming sensor data and make sleep stage prediction"""
        # Convert sensor data to DataFrame format expected by preprocessor
        data_point = pd.DataFrame({
            'timestamp': [current_time_minutes],
            'accelerometer_x': [sensor_data['accel_x']],
            'accelerometer_y': [sensor_data['accel_y']],
            'accelerometer_z': [sensor_data['accel_z']],
            'heart_rate': [sensor_data['heart_rate']],
            'sleep_stage': ['Unknown']
        })
        
        # Add derived features
        data_point['accel_magnitude'] = np.sqrt(
            data_point['accelerometer_x']**2 + 
            data_point['accelerometer_y']**2 + 
            data_point['accelerometer_z']**2
        )
        
        # Convert to spikes
        spike_data = self.preprocessor.convert_to_spikes(data_point)
        
        # Make prediction
        predicted_stage = self.snn_model.predict(spike_data)[0]
        
        # Store in history
        self.sleep_history.append({
            'time': current_time_minutes,
            'predicted_stage': predicted_stage,
            'sensor_data': sensor_data.copy()
        })
        
        # Check alarm conditions
        alarm_decision = self.check_alarm_conditions(predicted_stage, current_time_minutes)
        
        return predicted_stage, alarm_decision
    
    def check_alarm_conditions(self, current_stage, current_time):
        """Check if alarm should be triggered"""
        if self.alarm_triggered or self.target_wake_time is None:
            return {'trigger': False, 'reason': 'Already triggered or no alarm set'}
        
        # Calculate alarm window
        alarm_start = self.target_wake_time - self.alarm_window_minutes
        alarm_end = self.target_wake_time
        
        # Check if we're in the alarm window
        if current_time < alarm_start:
            return {'trigger': False, 'reason': 'Too early - not in alarm window'}
        
        if current_time > alarm_end:
            # Force alarm at target time regardless of sleep stage
            self.alarm_triggered = True
            return {
                'trigger': True, 
                'reason': f'Target time reached - waking from {current_stage} stage',
                'optimal': current_stage in self.optimal_stages
            }
        
        # Check if current stage is optimal for waking
        if current_stage in self.optimal_stages:
            # Additional check: make sure we've been in this stage for a few minutes
            if len(self.sleep_history) >= 3:
                recent_stages = [h['predicted_stage'] for h in self.sleep_history[-3:]]
                if recent_stages.count(current_stage) >= 2:  # Stable stage
                    self.alarm_triggered = True
                    return {
                        'trigger': True,
                        'reason': f'Optimal wake time - stable {current_stage} stage detected',
                        'optimal': True
                    }
        
        return {
            'trigger': False, 
            'reason': f'In alarm window but current stage is {current_stage} (waiting for optimal stage)'
        }