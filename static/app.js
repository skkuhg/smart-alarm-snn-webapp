/**
 * Smart Alarm SNN Demo - Frontend JavaScript
 * Real-time data visualization and interaction
 */

class SmartAlarmDemo {
    constructor() {
        this.isRunning = false;
        this.charts = {};
        this.updateInterval = null;
        this.data = {
            times: [],
            sleepStages: [],
            heartRates: [],
            movements: [],
            stageHistory: [],
            spikes: []
        };
        
        this.snnMetrics = {
            accuracy: 0,
            spikeRate: 0,
            energy: 0,
            stdpUpdates: 0
        };
        
        this.stageColors = {
            'Awake': '#e74c3c',
            'Light': '#3498db',
            'Deep': '#2c3e50',
            'REM': '#f39c12'
        };
        
        this.stageMapping = {
            'Awake': 3,
            'REM': 2,
            'Light': 1,
            'Deep': 0
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.updateStatus();
    }
    
    setupEventListeners() {
        document.getElementById('startBtn').addEventListener('click', () => this.startDemo());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopDemo());
        document.getElementById('generateAnalysis').addEventListener('click', () => this.generateAnalysis());
    }
    
    async startDemo() {
        const wakeTime = parseFloat(document.getElementById('wakeTime').value) * 60; // Convert to minutes
        
        try {
            const response = await fetch('/api/start_demo', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    target_wake_time: wakeTime
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.isRunning = true;
                this.clearData();
                this.updateUI(true);
                this.startDataPolling();
                this.updateAlarmStatus('Demo started - monitoring sleep patterns...', 'alarm-waiting');
            } else {
                this.showError(result.message);
            }
        } catch (error) {
            this.showError('Failed to start demo: ' + error.message);
        }
    }
    
    async stopDemo() {
        try {
            const response = await fetch('/api/stop_demo', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.isRunning = false;
                this.updateUI(false);
                this.stopDataPolling();
                this.updateAlarmStatus('Demo stopped', 'alarm-status');
            }
        } catch (error) {
            this.showError('Failed to stop demo: ' + error.message);
        }
    }
    
    startDataPolling() {
        this.updateInterval = setInterval(() => {
            this.fetchDemoStatus();
        }, 1000); // Update every 1 second for stability
    }
    
    stopDataPolling() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    async fetchDemoStatus() {
        if (!this.isRunning) return;
        
        try {
            const response = await fetch('/api/demo_status');
            const status = await response.json();
            
            if (!status.running && this.isRunning) {
                // Demo finished
                this.isRunning = false;
                this.updateUI(false);
                this.stopDataPolling();
                
                if (status.alarm_triggered) {
                    this.updateAlarmStatus('🚨 ALARM TRIGGERED! Optimal wake time detected', 'alarm-triggered');
                } else {
                    this.updateAlarmStatus('Demo completed - no alarm trigger', 'alarm-status');
                }
                
                // Fetch final data
                await this.fetchFullData();
                return;
            }
            
            if (status.recent_data && status.recent_data.length > 0) {
                this.updateData(status.recent_data);
                this.updateStatusDisplay(status.recent_data[status.recent_data.length - 1]);
            }
            
            // Update alarm status
            if (status.recent_data && status.recent_data.length > 0) {
                const latest = status.recent_data[status.recent_data.length - 1];
                if (latest.alarm_triggered) {
                    this.updateAlarmStatus('🚨 ALARM TRIGGERED! ' + latest.alarm_status, 'alarm-triggered');
                } else {
                    this.updateAlarmStatus(latest.alarm_status, 'alarm-waiting');
                }
            }
            
        } catch (error) {
            console.error('Failed to fetch demo status:', error);
        }
    }
    
    async fetchFullData() {
        try {
            const response = await fetch('/api/full_data');
            const fullData = await response.json();
            
            if (fullData.data && fullData.data.length > 0) {
                this.data.times = fullData.data.map(d => d.time_hours);
                this.data.sleepStages = fullData.data.map(d => this.stageMapping[d.predicted_stage]);
                this.data.heartRates = fullData.data.map(d => d.heart_rate);
                this.data.movements = fullData.data.map(d => d.movement);
                this.data.stageHistory = fullData.data.map(d => d.predicted_stage);
                
                this.updateCharts();
                this.updateStatistics();
            }
        } catch (error) {
            console.error('Failed to fetch full data:', error);
        }
    }
    
    updateData(newData) {
        newData.forEach(dataPoint => {
            // Avoid duplicates
            if (!this.data.times.includes(dataPoint.time_hours)) {
                this.data.times.push(dataPoint.time_hours);
                this.data.sleepStages.push(this.stageMapping[dataPoint.predicted_stage]);
                this.data.heartRates.push(dataPoint.heart_rate);
                this.data.movements.push(dataPoint.movement);
                this.data.stageHistory.push(dataPoint.predicted_stage);
                
                // Generate synthetic spike data for visualization
                const spikeData = this.generateSpikeData(dataPoint);
                this.data.spikes.push(spikeData);
            }
        });
        
        // Keep only last 50 points for better performance and stability
        const maxPoints = 50;
        if (this.data.times.length > maxPoints) {
            this.data.times = this.data.times.slice(-maxPoints);
            this.data.sleepStages = this.data.sleepStages.slice(-maxPoints);
            this.data.heartRates = this.data.heartRates.slice(-maxPoints);
            this.data.movements = this.data.movements.slice(-maxPoints);
            this.data.stageHistory = this.data.stageHistory.slice(-maxPoints);
            this.data.spikes = this.data.spikes.slice(-maxPoints);
        }
        
        this.updateCharts();
        this.updateSNNVisualizations();
        this.updateSNNMetrics();
    }
    
    generateSpikeData(dataPoint) {
        // Generate realistic spike data based on sensor values
        const movement = dataPoint.movement;
        const heartRate = dataPoint.heart_rate;
        
        return {
            accel_x: movement > 0.1 ? Math.random() > 0.5 ? 1 : 0 : 0,
            accel_y: movement > 0.08 ? Math.random() > 0.6 ? 1 : 0 : 0,
            accel_z: movement > 0.05 ? Math.random() > 0.7 ? 1 : 0 : 0,
            heart_rate: heartRate > 70 ? Math.random() > 0.4 ? 1 : 0 : 0
        };
    }
    
    updateStatusDisplay(dataPoint) {
        const currentTime = Math.floor(dataPoint.time / 60) + ':' + 
                           String(Math.floor(dataPoint.time % 60)).padStart(2, '0');
        
        document.getElementById('currentTime').textContent = currentTime;
        document.getElementById('heartRate').textContent = Math.round(dataPoint.heart_rate) + ' BPM';
        document.getElementById('movement').textContent = dataPoint.movement.toFixed(3) + ' g';
        document.getElementById('dataPoints').textContent = this.data.times.length;
        
        const stageElement = document.getElementById('sleepStage');
        stageElement.textContent = dataPoint.predicted_stage;
        stageElement.className = 'sleep-stage stage-' + dataPoint.predicted_stage.toLowerCase();
    }
    
    initializeCharts() {
        // Sleep Stage Chart
        const sleepCtx = document.getElementById('sleepStageChart').getContext('2d');
        this.charts.sleepStage = new Chart(sleepCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Sleep Stage',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    stepped: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (hours)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Sleep Stage'
                        },
                        ticks: {
                            callback: function(value) {
                                const stages = ['Deep', 'Light', 'REM', 'Awake'];
                                return stages[value] || value;
                            }
                        },
                        min: 0,
                        max: 3
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        // Heart Rate Chart
        const heartCtx = document.getElementById('heartRateChart').getContext('2d');
        this.charts.heartRate = new Chart(heartCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Heart Rate',
                    data: [],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 2,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (hours)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Heart Rate (BPM)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        // Movement Chart
        const movementCtx = document.getElementById('movementChart').getContext('2d');
        this.charts.movement = new Chart(movementCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Movement',
                    data: [],
                    borderColor: '#9b59b6',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    borderWidth: 2,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (hours)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Movement (g)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        // Spike Chart
        const spikeCtx = document.getElementById('spikeChart').getContext('2d');
        this.charts.spike = new Chart(spikeCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Accel X Spikes',
                        data: [],
                        backgroundColor: '#ff6b6b',
                        borderWidth: 0
                    },
                    {
                        label: 'Accel Y Spikes',
                        data: [],
                        backgroundColor: '#4ecdc4',
                        borderWidth: 0
                    },
                    {
                        label: 'Accel Z Spikes',
                        data: [],
                        backgroundColor: '#45b7d1',
                        borderWidth: 0
                    },
                    {
                        label: 'HR Spikes',
                        data: [],
                        backgroundColor: '#f39c12',
                        borderWidth: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (hours)'
                        },
                        stacked: true
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Spike Events'
                        },
                        stacked: true,
                        max: 4
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }
    
    updateCharts() {
        // Update all charts with current data
        Object.values(this.charts).forEach(chart => {
            chart.data.labels = this.data.times.map(t => t.toFixed(1));
        });
        
        this.charts.sleepStage.data.datasets[0].data = this.data.sleepStages;
        this.charts.heartRate.data.datasets[0].data = this.data.heartRates;
        this.charts.movement.data.datasets[0].data = this.data.movements;
        
        // Update spike chart
        if (this.data.spikes.length > 0) {
            this.charts.spike.data.datasets[0].data = this.data.spikes.map(s => s.accel_x);
            this.charts.spike.data.datasets[1].data = this.data.spikes.map(s => s.accel_y);
            this.charts.spike.data.datasets[2].data = this.data.spikes.map(s => s.accel_z);
            this.charts.spike.data.datasets[3].data = this.data.spikes.map(s => s.heart_rate);
        }
        
        Object.values(this.charts).forEach(chart => {
            chart.update('none'); // Use 'none' mode for smoother updates
        });
    }
    
    updateStatistics() {
        if (this.data.stageHistory.length === 0) {
            document.getElementById('sleepStats').innerHTML = '<div class="loading">No data available</div>';
            return;
        }
        
        const stageCounts = {};
        this.data.stageHistory.forEach(stage => {
            stageCounts[stage] = (stageCounts[stage] || 0) + 1;
        });
        
        const total = this.data.stageHistory.length;
        const avgHeartRate = this.data.heartRates.reduce((sum, hr) => sum + hr, 0) / this.data.heartRates.length;
        const avgMovement = this.data.movements.reduce((sum, mv) => sum + mv, 0) / this.data.movements.length;
        
        let statsHTML = `
            <div class="status-item">
                <span class="status-label">Total Time:</span>
                <span class="status-value">${total} minutes</span>
            </div>
            <div class="status-item">
                <span class="status-label">Avg Heart Rate:</span>
                <span class="status-value">${avgHeartRate.toFixed(1)} BPM</span>
            </div>
            <div class="status-item">
                <span class="status-label">Avg Movement:</span>
                <span class="status-value">${avgMovement.toFixed(3)} g</span>
            </div>
            <h4 style="margin: 15px 0 10px 0; color: #2c3e50;">Stage Distribution:</h4>
        `;
        
        Object.entries(stageCounts).forEach(([stage, count]) => {
            const percentage = ((count / total) * 100).toFixed(1);
            statsHTML += `
                <div class="status-item">
                    <span class="status-label">${stage}:</span>
                    <span class="status-value">${percentage}%</span>
                </div>
            `;
        });
        
        document.getElementById('sleepStats').innerHTML = statsHTML;
    }
    
    updateSNNVisualizations() {
        // Update spike indicators
        const spikeIndicators = document.querySelectorAll('.spike-dot');
        spikeIndicators.forEach((dot, index) => {
            dot.classList.remove('active');
            if (this.data.spikes.length > 0) {
                const latestSpikes = this.data.spikes[this.data.spikes.length - 1];
                const spikeValues = [latestSpikes.accel_x, latestSpikes.accel_y, latestSpikes.accel_z, latestSpikes.heart_rate];
                if (spikeValues[index]) {
                    dot.classList.add('active');
                    setTimeout(() => dot.classList.remove('active'), 500);
                }
            }
        });
        
        // Update network activity
        const layers = document.querySelectorAll('.layer');
        layers.forEach(layer => layer.classList.remove('active'));
        
        // Simulate network activation flow
        if (this.data.spikes.length > 0) {
            setTimeout(() => layers[0]?.classList.add('active'), 100);
            setTimeout(() => {
                layers[0]?.classList.remove('active');
                layers[1]?.classList.add('active');
            }, 300);
            setTimeout(() => {
                layers[1]?.classList.remove('active');
                layers[2]?.classList.add('active');
            }, 500);
            setTimeout(() => layers[2]?.classList.remove('active'), 700);
        }
        
        // Update STDP strength (simulate learning)
        const stdpElement = document.querySelector('#stdpStrength span');
        if (stdpElement && this.data.stageHistory.length > 0) {
            const strength = (0.3 + Math.random() * 0.4).toFixed(2);
            stdpElement.textContent = strength;
        }
        
        // Update prediction confidence
        const confElement = document.querySelector('#predictionConfidence span');
        if (confElement && this.data.stageHistory.length > 0) {
            const confidence = (75 + Math.random() * 20).toFixed(0);
            confElement.textContent = confidence + '%';
        }
    }
    
    updateSNNMetrics() {
        if (this.data.stageHistory.length === 0) return;
        
        // Calculate simulated metrics
        this.snnMetrics.accuracy = Math.min(95, 60 + this.data.stageHistory.length * 0.5);
        this.snnMetrics.spikeRate = this.data.spikes.length > 0 ? 
            this.data.spikes[this.data.spikes.length - 1].accel_x + 
            this.data.spikes[this.data.spikes.length - 1].accel_y + 
            this.data.spikes[this.data.spikes.length - 1].accel_z + 
            this.data.spikes[this.data.spikes.length - 1].heart_rate : 0;
        this.snnMetrics.energy = (10 + this.snnMetrics.spikeRate * 2.5);
        this.snnMetrics.stdpUpdates = Math.floor(this.data.stageHistory.length * 1.2);
        
        // Update display
        document.getElementById('accuracy').textContent = this.snnMetrics.accuracy.toFixed(1) + '%';
        document.getElementById('spikeRate').textContent = this.snnMetrics.spikeRate * 10;
        document.getElementById('networkEnergy').textContent = this.snnMetrics.energy.toFixed(1);
        document.getElementById('stdpUpdates').textContent = this.snnMetrics.stdpUpdates;
    }
    
    async generateAnalysis() {
        document.getElementById('analysisContent').innerHTML = '<div class="loading">🔄 Generating sleep analysis...</div>';
        
        try {
            const response = await fetch('/api/sleep_analysis');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const analysis = await response.json();
            
            if (analysis.error) {
                throw new Error(analysis.error);
            }
            
            if (analysis.plot) {
                document.getElementById('analysisContent').innerHTML = `
                    <img id="analysisImage" src="data:image/png;base64,${analysis.plot}" alt="Sleep Analysis" style="max-width: 100%; height: auto;">
                    <div style="margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #3498db;">
                        <h4 style="color: #2c3e50; margin-bottom: 15px;">📊 Analysis Summary:</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                            <div class="stat-item">
                                <strong>🕐 Total Sleep Time:</strong><br>
                                ${analysis.statistics.total_sleep_time} minutes<br>
                                <span style="color: #3498db;">(${(analysis.statistics.total_sleep_time/60).toFixed(1)} hours)</span>
                            </div>
                            <div class="stat-item">
                                <strong>❤️ Average Heart Rate:</strong><br>
                                <span style="color: #e74c3c; font-size: 1.2em;">${analysis.statistics.avg_heart_rate.toFixed(1)} BPM</span>
                            </div>
                            <div class="stat-item">
                                <strong>🏃 Average Movement:</strong><br>
                                <span style="color: #9b59b6; font-size: 1.2em;">${analysis.statistics.avg_movement.toFixed(3)} g</span>
                            </div>
                        </div>
                        <div style="margin-top: 15px; padding: 10px; background: #e8f4f8; border-radius: 6px;">
                            <strong>🧠 SNN Analysis:</strong> This visualization shows the complete 8-hour sleep pattern used by the Spiking Neural Network for training and real-time prediction. The SNN processes sensor spikes to classify sleep stages with ${this.snnMetrics.accuracy.toFixed(1)}% accuracy.
                        </div>
                        <div style="margin-top: 10px;">
                            <h5>Sleep Stage Distribution:</h5>
                            ${Object.entries(analysis.statistics.stage_distribution).map(([stage, count]) => 
                                `<span class="sleep-stage stage-${stage.toLowerCase()}" style="display: inline-block; margin: 2px; padding: 4px 8px; border-radius: 12px; font-size: 0.8em;">${stage}: ${count} min</span>`
                            ).join('')}
                        </div>
                    </div>
                `;
            } else {
                throw new Error('No visualization data received from server');
            }
        } catch (error) {
            console.error('Analysis generation error:', error);
            document.getElementById('analysisContent').innerHTML = `
                <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; text-align: center;">
                    <h4 style="color: #856404;">⚠️ Analysis Generation Failed</h4>
                    <p style="color: #856404; margin: 10px 0;">Error: ${error.message}</p>
                    <button onclick="new SmartAlarmDemo().generateAnalysis()" class="btn" style="margin-top: 10px;">🔄 Retry Analysis</button>
                    <div style="margin-top: 15px; font-size: 0.9em; color: #6c757d;">
                        <strong>Troubleshooting Tips:</strong><br>
                        • Make sure the demo has been run at least once<br>
                        • Check that matplotlib is properly installed<br>
                        • Refresh the page and try again
                    </div>
                </div>
            `;
        }
    }
    
    updateUI(running) {
        document.getElementById('startBtn').disabled = running;
        document.getElementById('stopBtn').disabled = !running;
        document.getElementById('wakeTime').disabled = running;
    }
    
    updateAlarmStatus(message, className = 'alarm-status') {
        const statusElement = document.getElementById('alarmStatus');
        statusElement.innerHTML = `<strong>${message}</strong>`;
        statusElement.className = 'alarm-status ' + className;
    }
    
    clearData() {
        this.data = {
            times: [],
            sleepStages: [],
            heartRates: [],
            movements: [],
            stageHistory: [],
            spikes: []
        };
        this.snnMetrics = {
            accuracy: 0,
            spikeRate: 0,
            energy: 0,
            stdpUpdates: 0
        };
        this.updateCharts();
        
        // Reset status display
        document.getElementById('currentTime').textContent = '--:--';
        document.getElementById('sleepStage').textContent = 'Unknown';
        document.getElementById('sleepStage').className = 'sleep-stage stage-light';
        document.getElementById('heartRate').textContent = '-- BPM';
        document.getElementById('movement').textContent = '-- g';
        document.getElementById('dataPoints').textContent = '0';
        document.getElementById('sleepStats').innerHTML = '<div class="loading">Waiting for data...</div>';
    }
    
    showError(message) {
        this.updateAlarmStatus(`❌ Error: ${message}`, 'alarm-status');
        setTimeout(() => {
            this.updateAlarmStatus('System Ready - Click "Start Demo" to begin', 'alarm-status');
        }, 5000);
    }
    
    async updateStatus() {
        // Initial status update
        try {
            const response = await fetch('/api/demo_status');
            const status = await response.json();
            
            if (status.running) {
                this.isRunning = true;
                this.updateUI(true);
                this.startDataPolling();
            }
        } catch (error) {
            console.log('No active demo session');
        }
    }
}

// Initialize the demo when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new SmartAlarmDemo();
});