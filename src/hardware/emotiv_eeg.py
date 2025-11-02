"""
Emotiv EEG Hardware Interface Module

This module handles communication with Emotiv EEG devices for capturing
brainwave data in real-time.
"""

import time
import threading
import queue
import logging
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass
import json

try:
    import cortex
except ImportError:
    cortex = None
    logging.warning("Cortex Python SDK not available. Install with: pip install cortex-python")


@dataclass
class EEGSample:
    """Data structure for a single EEG sample"""
    timestamp: float
    session_id: str
    channels: Dict[str, float]  # Channel name -> value mapping
    quality: Dict[str, int]     # Channel quality indicators
    raw_data: Dict             # Raw data from device


class EmotivEEG:
    """Interface class for Emotiv EEG devices using Cortex API"""
    
    def __init__(self, client_id: str, client_secret: str, license: str, 
                 headset_id: str = "auto"):
        """
        Initialize Emotiv EEG interface
        
        Args:
            client_id: Emotiv client ID
            client_secret: Emotiv client secret
            license: Emotiv license key
            headset_id: Specific headset ID or "auto" for auto-detection
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.license = license
        self.headset_id = headset_id
        
        self.cortex_client = None
        self.auth_token = None
        self.session_id = None
        self.is_connected = False
        self.is_streaming = False
        
        # Data collection
        self.data_queue = queue.Queue(maxsize=1000)
        self.data_callbacks: List[Callable[[EEGSample], None]] = []
        
        # Threading
        self.stream_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Channel mappings for different Emotiv devices
        self.channel_mappings = {
            'EPOC+': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
            'EPOC X': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
            'INSIGHT': ['AF3', 'AF4', 'T7', 'T8', 'Pz']
        }
        
    def initialize_cortex(self) -> bool:
        """Initialize Cortex client and authenticate"""
        if cortex is None:
            self.logger.error("Cortex Python SDK not available")
            return False
            
        try:
            # Initialize Cortex client
            self.cortex_client = cortex.Cortex(self.client_id, self.client_secret, debug=False)
            
            # Connect to Cortex service
            self.cortex_client.open()
            
            # Authenticate
            response = self.cortex_client.get_user_login()
            if not response:
                # Request access
                self.cortex_client.request_access()
                
            # Get auth token
            auth_response = self.cortex_client.authorize()
            self.auth_token = auth_response.get('cortexToken')
            
            if not self.auth_token:
                self.logger.error("Failed to get authentication token")
                return False
                
            self.logger.info("Cortex client initialized and authenticated")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Cortex: {e}")
            return False
            
    def detect_headsets(self) -> List[Dict]:
        """
        Detect available Emotiv headsets
        
        Returns:
            List of headset information dictionaries
        """
        if not self.cortex_client:
            return []
            
        try:
            headsets = self.cortex_client.query_headsets()
            available_headsets = []
            
            for headset in headsets:
                if headset.get('status') == 'connected':
                    available_headsets.append({
                        'id': headset.get('id'),
                        'status': headset.get('status'),
                        'connectedBy': headset.get('connectedBy'),
                        'firmware': headset.get('firmware'),
                        'model': headset.get('model', 'Unknown')
                    })
                    
            return available_headsets
            
        except Exception as e:
            self.logger.error(f"Error detecting headsets: {e}")
            return []
            
    def connect(self) -> bool:
        """
        Connect to Emotiv EEG device
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.is_connected:
            return True
            
        # Initialize Cortex if needed
        if not self.cortex_client:
            if not self.initialize_cortex():
                return False
                
        try:
            # Detect headsets
            headsets = self.detect_headsets()
            if not headsets:
                self.logger.error("No connected Emotiv headsets found")
                return False
                
            # Select headset
            if self.headset_id == "auto":
                selected_headset = headsets[0]
                self.headset_id = selected_headset['id']
            else:
                selected_headset = next((h for h in headsets if h['id'] == self.headset_id), None)
                if not selected_headset:
                    self.logger.error(f"Headset {self.headset_id} not found")
                    return False
                    
            self.logger.info(f"Using headset: {selected_headset}")
            
            # Create session
            session_response = self.cortex_client.create_session(
                cortex_token=self.auth_token,
                headset=self.headset_id,
                status='active'
            )
            
            self.session_id = session_response.get('id')
            if not self.session_id:
                self.logger.error("Failed to create session")
                return False
                
            self.is_connected = True
            self.logger.info(f"Connected to Emotiv EEG, session: {self.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Emotiv EEG: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from Emotiv EEG device"""
        self.stop_streaming()
        
        if self.cortex_client and self.session_id:
            try:
                self.cortex_client.update_session(
                    cortex_token=self.auth_token,
                    session=self.session_id,
                    status='close'
                )
            except Exception as e:
                self.logger.error(f"Error closing session: {e}")
                
        if self.cortex_client:
            try:
                self.cortex_client.close()
            except Exception as e:
                self.logger.error(f"Error closing Cortex client: {e}")
                
        self.is_connected = False
        self.session_id = None
        self.auth_token = None
        self.logger.info("Disconnected from Emotiv EEG")
        
    def start_streaming(self, streams: List[str] = None):
        """
        Start EEG data streaming
        
        Args:
            streams: List of stream types to subscribe to
                    ['eeg', 'mot', 'dev', 'met', 'pow', 'fac', 'com']
        """
        if self.is_streaming:
            return
            
        if not self.is_connected:
            if not self.connect():
                return
                
        if streams is None:
            streams = ['eeg', 'met']  # EEG data and contact quality
            
        try:
            # Subscribe to data streams
            for stream in streams:
                self.cortex_client.sub_request({
                    'cortexToken': self.auth_token,
                    'session': self.session_id,
                    'streams': [stream]
                })
                
            # Set up data callback
            self.cortex_client.bind(new_eeg_data=self._eeg_data_callback)
            self.cortex_client.bind(new_met_data=self._met_data_callback)
            
            self.stop_event.clear()
            self.is_streaming = True
            
            # Start background thread for processing
            self.stream_thread = threading.Thread(
                target=self._streaming_worker,
                daemon=True
            )
            self.stream_thread.start()
            
            self.logger.info(f"Started EEG streaming: {streams}")
            
        except Exception as e:
            self.logger.error(f"Failed to start EEG streaming: {e}")
            
    def stop_streaming(self):
        """Stop EEG data streaming"""
        if not self.is_streaming:
            return
            
        self.stop_event.set()
        self.is_streaming = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
            
        try:
            if self.cortex_client and self.session_id:
                self.cortex_client.unsub_request({
                    'cortexToken': self.auth_token,
                    'session': self.session_id,
                    'streams': ['eeg', 'met']
                })
        except Exception as e:
            self.logger.error(f"Error stopping EEG streaming: {e}")
            
        self.logger.info("Stopped EEG streaming")
        
    def _eeg_data_callback(self, data):
        """Callback for EEG data"""
        if data and 'eeg' in data:
            eeg_data = data['eeg']
            timestamp = time.time()
            
            # Parse EEG data
            if len(eeg_data) > 2:
                channels = {}
                # Map channel data (structure depends on device)
                channel_names = self._get_channel_names()
                for i, value in enumerate(eeg_data[2:]):  # Skip timestamp and counter
                    if i < len(channel_names):
                        channels[channel_names[i]] = float(value)
                        
                sample = EEGSample(
                    timestamp=timestamp,
                    session_id=self.session_id,
                    channels=channels,
                    quality={},  # Will be filled by met data
                    raw_data=data
                )
                
                self._process_sample(sample)
                
    def _met_data_callback(self, data):
        """Callback for contact quality data"""
        # Contact quality data processing
        pass
        
    def _get_channel_names(self) -> List[str]:
        """Get channel names for the current device"""
        # This would be determined by the specific headset model
        return self.channel_mappings.get('EPOC+', [])  # Default to EPOC+
        
    def _process_sample(self, sample: EEGSample):
        """Process a new EEG sample"""
        # Add to queue
        try:
            self.data_queue.put_nowait(sample)
        except queue.Full:
            # Remove oldest sample if queue is full
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(sample)
            except queue.Empty:
                pass
                
        # Call registered callbacks
        for callback in self.data_callbacks:
            try:
                callback(sample)
            except Exception as e:
                self.logger.error(f"Error in EEG callback: {e}")
                
    def _streaming_worker(self):
        """Background worker for data processing"""
        while not self.stop_event.is_set():
            time.sleep(0.01)  # Small delay to prevent busy waiting
            
    def get_latest_samples(self, max_samples: int = 100) -> List[EEGSample]:
        """
        Get latest EEG samples from the queue
        
        Args:
            max_samples: Maximum number of samples to return
            
        Returns:
            List of recent EEGSample objects
        """
        samples = []
        for _ in range(min(max_samples, self.data_queue.qsize())):
            try:
                samples.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return samples
        
    def add_data_callback(self, callback: Callable[[EEGSample], None]):
        """Add callback function for real-time data processing"""
        self.data_callbacks.append(callback)
        
    def remove_data_callback(self, callback: Callable[[EEGSample], None]):
        """Remove callback function"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
            
    def get_device_info(self) -> Dict:
        """Get information about the connected device"""
        if not self.is_connected:
            return {}
            
        try:
            headsets = self.detect_headsets()
            current_headset = next((h for h in headsets if h['id'] == self.headset_id), {})
            return current_headset
        except Exception as e:
            self.logger.error(f"Error getting device info: {e}")
            return {}
            
    def check_connection_status(self) -> bool:
        """Check if device is actually connected and responding"""
        if not self.is_connected or not self.cortex_client:
            return False
            
        try:
            # Try to query headsets to verify connection
            headsets = self.detect_headsets()
            current_headset = next((h for h in headsets if h['id'] == self.headset_id), None)
            
            if current_headset and current_headset.get('status') == 'connected':
                return True
            else:
                # Connection lost, update status
                self.is_connected = False
                self.is_streaming = False
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking connection status: {e}")
            self.is_connected = False
            self.is_streaming = False
            return False


# Mock EEG interface for testing without hardware
class MockEmotivEEG(EmotivEEG):
    """Mock EEG interface for testing without hardware"""
    
    def __init__(self, simulate_disconnected: bool = False):
        """Initialize mock EEG interface
        
        Args:
            simulate_disconnected: If True, simulates a disconnected device
        """
        self.is_connected = False
        self.is_streaming = False
        self.session_id = "mock_session"
        self.data_queue = queue.Queue(maxsize=1000)
        self.data_callbacks = []
        self.stop_event = threading.Event()
        self.stream_thread = None
        self.logger = logging.getLogger(__name__)
        self.simulate_disconnected = simulate_disconnected
        self.connection_attempts = 0
        
    def connect(self) -> bool:
        """Mock connection with realistic failure simulation"""
        self.connection_attempts += 1
        
        if self.simulate_disconnected:
            # Simulate various connection failure scenarios
            if self.connection_attempts <= 3:
                self.logger.warning(f"Mock EEG connection attempt {self.connection_attempts} failed (simulated)")
                return False
            else:
                # Eventually succeed after multiple attempts
                self.is_connected = True
                self.logger.info("Connected to mock Emotiv EEG (after retries)")
                return True
        else:
            # Normal successful connection
            self.is_connected = True
            self.logger.info("Connected to mock Emotiv EEG")
            return True
        
    def disconnect(self):
        """Mock disconnection"""
        self.stop_streaming()
        self.is_connected = False
        self.logger.info("Disconnected from mock Emotiv EEG")
        
    def start_streaming(self, streams: List[str] = None):
        """Start mock data streaming"""
        if self.is_streaming:
            return
            
        self.stop_event.clear()
        self.is_streaming = True
        
        self.stream_thread = threading.Thread(
            target=self._mock_streaming_worker,
            daemon=True
        )
        self.stream_thread.start()
        
        self.logger.info("Started mock EEG streaming")
        
    def _mock_streaming_worker(self):
        """Generate mock EEG data"""
        import random
        import math
        
        channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        
        while not self.stop_event.is_set():
            timestamp = time.time()
            
            # Generate realistic EEG-like data (microvolts)
            channels = {}
            for channel in channel_names:
                # Simulate brain waves with noise
                alpha = 10 * math.sin(2 * math.pi * 10 * timestamp)  # 10 Hz alpha wave
                beta = 5 * math.sin(2 * math.pi * 20 * timestamp)    # 20 Hz beta wave
                noise = random.gauss(0, 2)
                channels[channel] = alpha + beta + noise
                
            sample = EEGSample(
                timestamp=timestamp,
                session_id=self.session_id,
                channels=channels,
                quality={ch: random.randint(0, 4) for ch in channel_names},
                raw_data={'mock': True}
            )
            
            self._process_sample(sample)
            time.sleep(1.0 / 128.0)  # 128 Hz sampling rate
            
    def check_connection_status(self) -> bool:
        """Check mock connection status with realistic simulation"""
        if self.simulate_disconnected:
            # Simulate intermittent connection issues
            import random
            if random.random() < 0.1:  # 10% chance of temporary disconnection
                self.is_connected = False
                self.logger.warning("Mock EEG device temporarily disconnected")
                return False
        
        return self.is_connected


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Use mock EEG for testing
    eeg = MockEmotivEEG()
    
    if eeg.connect():
        print("Connected to EEG!")
        
        def print_callback(sample: EEGSample):
            af3_value = sample.channels.get('AF3', 0)
            print(f"EEG: AF3={af3_value:.2f} at {sample.timestamp}")
            
        eeg.add_data_callback(print_callback)
        eeg.start_streaming()
        
        time.sleep(3)
        
        eeg.stop_streaming()
        
        # Get samples
        samples = eeg.get_latest_samples(10)
        print(f"Collected {len(samples)} samples")
        
        eeg.disconnect()
    else:
        print("Failed to connect to EEG")