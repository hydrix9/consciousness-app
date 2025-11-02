"""
Enhanced Mock EEG Implementation

Provides realistic EEG device simulation with configurable behaviors for comprehensive testing.
Eliminates false positive connection status issues.
"""

import time
import random
import math
import threading
import queue
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .eeg_bridge import EEGSource, EEGPacket, ConnectionInfo, ConnectionStatus, EEGSourceType


class EEGSimulationMode:
    """Simulation modes for different testing scenarios"""
    STABLE = "stable"           # Reliable connection, good signal quality
    UNSTABLE = "unstable"       # Intermittent disconnections, variable quality
    DISCONNECTED = "disconnected"  # Always fails to connect
    CONNECTING = "connecting"   # Gets stuck in connecting state
    SLOW_CONNECT = "slow_connect"  # Takes a long time to connect
    POOR_SIGNAL = "poor_signal"    # Connected but poor signal quality


@dataclass
class MockDeviceProfile:
    """Profile for different mock device behaviors"""
    name: str
    connection_success_rate: float  # 0.0-1.0
    connection_delay_range: tuple   # (min, max) seconds
    signal_quality_range: tuple    # (min, max) 0.0-1.0
    disconnect_probability: float   # Per-second probability
    data_dropout_rate: float       # Probability of missing data packets
    noise_level: float             # Signal noise multiplier


# Predefined device profiles
DEVICE_PROFILES = {
    EEGSimulationMode.STABLE: MockDeviceProfile(
        name="Stable Device",
        connection_success_rate=1.0,
        connection_delay_range=(0.5, 1.0),
        signal_quality_range=(0.8, 0.95),
        disconnect_probability=0.0,
        data_dropout_rate=0.0,
        noise_level=0.02
    ),
    EEGSimulationMode.UNSTABLE: MockDeviceProfile(
        name="Unstable Device",
        connection_success_rate=0.7,
        connection_delay_range=(1.0, 3.0),
        signal_quality_range=(0.3, 0.8),
        disconnect_probability=0.02,  # 2% chance per second
        data_dropout_rate=0.05,
        noise_level=0.1
    ),
    EEGSimulationMode.DISCONNECTED: MockDeviceProfile(
        name="Disconnected Device",
        connection_success_rate=0.0,
        connection_delay_range=(2.0, 5.0),
        signal_quality_range=(0.0, 0.0),
        disconnect_probability=1.0,
        data_dropout_rate=1.0,
        noise_level=1.0
    ),
    EEGSimulationMode.CONNECTING: MockDeviceProfile(
        name="Connecting Device",
        connection_success_rate=0.0,  # Never completes connection
        connection_delay_range=(10.0, 30.0),
        signal_quality_range=(0.0, 0.0),
        disconnect_probability=0.0,
        data_dropout_rate=1.0,
        noise_level=1.0
    ),
    EEGSimulationMode.SLOW_CONNECT: MockDeviceProfile(
        name="Slow Connect Device",
        connection_success_rate=0.9,
        connection_delay_range=(5.0, 10.0),
        signal_quality_range=(0.7, 0.9),
        disconnect_probability=0.01,
        data_dropout_rate=0.02,
        noise_level=0.05
    ),
    EEGSimulationMode.POOR_SIGNAL: MockDeviceProfile(
        name="Poor Signal Device",
        connection_success_rate=0.95,
        connection_delay_range=(0.5, 1.5),
        signal_quality_range=(0.1, 0.4),
        disconnect_probability=0.005,
        data_dropout_rate=0.1,
        noise_level=0.3
    )
}


class EnhancedMockEEG(EEGSource):
    """Enhanced mock EEG with realistic device simulation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration
        self.mode = config.get("mode", EEGSimulationMode.STABLE)
        self.profile = DEVICE_PROFILES.get(self.mode, DEVICE_PROFILES[EEGSimulationMode.STABLE])
        
        # Device simulation
        self.device_id = f"MOCK-{random.randint(1000, 9999)}"
        self.session_id = None
        self.connection_start_time = None
        self.last_connection_attempt = 0
        self.connection_attempts = 0
        
        # Data generation
        self.channels = config.get("channels", [
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", 
            "P8", "T8", "FC6", "F4", "F8", "AF4"
        ])
        self.sampling_rate = config.get("sampling_rate", 128)  # Hz
        self.data_queue = queue.Queue(maxsize=256)
        
        # Threading
        self._streaming_thread: Optional[threading.Thread] = None
        self._stop_streaming = threading.Event()
        
        # Status tracking
        self._last_data_time = 0.0
        self._packets_generated = 0
        self._connection_quality = 0.0
        self._current_signal_quality = 0.0
        
        # Realistic timing
        self._start_time = time.time()
        
        self.logger.info(f"Mock EEG initialized: mode={self.mode}, profile={self.profile.name}")
        
    async def connect(self) -> bool:
        """Simulate device connection with realistic delays and failures"""
        self.connection_attempts += 1
        self.last_connection_attempt = time.time()
        
        self.logger.info(f"Mock EEG connection attempt #{self.connection_attempts} (mode: {self.mode})")
        
        # Update status to connecting
        self._status = ConnectionStatus.CONNECTING
        self._error_message = None
        
        # Simulate connection delay
        delay = random.uniform(*self.profile.connection_delay_range)
        await self._async_delay(delay)
        
        # Check if connection should succeed
        if random.random() > self.profile.connection_success_rate:
            self._status = ConnectionStatus.ERROR
            self._error_message = f"Mock connection failed (attempt {self.connection_attempts})"
            self.logger.warning(self._error_message)
            return False
            
        # Special handling for connecting mode (never completes)
        if self.mode == EEGSimulationMode.CONNECTING:
            self.logger.info("Mock EEG stuck in connecting state (simulated)")
            # Stay in CONNECTING status indefinitely
            return False
            
        # Successful connection
        self.connection_start_time = time.time()
        self.session_id = f"mock_session_{int(self.connection_start_time)}"
        self._status = ConnectionStatus.CONNECTED
        self._connection_quality = random.uniform(*self.profile.signal_quality_range)
        
        self.logger.info(f"Mock EEG connected successfully: device_id={self.device_id}, session_id={self.session_id}")
        return True
        
    async def disconnect(self):
        """Simulate device disconnection"""
        await self.stop_streaming()
        
        self._status = ConnectionStatus.DISCONNECTED
        self.session_id = None
        self.connection_start_time = None
        self._error_message = None
        
        self.logger.info("Mock EEG disconnected")
        
    async def start_streaming(self) -> bool:
        """Start mock data streaming"""
        if self._status != ConnectionStatus.CONNECTED:
            self.logger.warning("Cannot start streaming: not connected")
            return False
            
        if self._streaming_thread and self._streaming_thread.is_alive():
            return True  # Already streaming
            
        self._stop_streaming.clear()
        self._streaming_thread = threading.Thread(
            target=self._data_generation_worker,
            name="MockEEGStream",
            daemon=True
        )
        self._streaming_thread.start()
        
        self._status = ConnectionStatus.STREAMING
        self.logger.info("Mock EEG streaming started")
        return True
        
    async def stop_streaming(self):
        """Stop mock data streaming"""
        self._stop_streaming.set()
        
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=2.0)
            
        if self._status == ConnectionStatus.STREAMING:
            self._status = ConnectionStatus.CONNECTED
            
        self.logger.info("Mock EEG streaming stopped")
        
    def _data_generation_worker(self):
        """Background thread for generating realistic EEG data"""
        self.logger.info(f"Mock EEG data generation started: {self.sampling_rate}Hz, {len(self.channels)} channels")
        
        last_packet_time = time.time()
        packet_interval = 1.0 / self.sampling_rate
        
        while not self._stop_streaming.is_set():
            try:
                current_time = time.time()
                
                # Check for simulated disconnection
                if self._should_disconnect():
                    self.logger.warning("Mock EEG simulated disconnection")
                    self._status = ConnectionStatus.ERROR
                    self._error_message = "Simulated device disconnection"
                    break
                    
                # Check if it's time for next packet
                if current_time - last_packet_time >= packet_interval:
                    # Skip packet if simulating data dropout
                    if random.random() < self.profile.data_dropout_rate:
                        last_packet_time = current_time
                        continue
                        
                    packet = self._generate_eeg_packet(current_time)
                    
                    # Add to queue
                    try:
                        self.data_queue.put_nowait(packet)
                        self._notify_callbacks(packet)
                        
                        self._packets_generated += 1
                        self._last_data_time = current_time
                        last_packet_time = current_time
                        
                    except queue.Full:
                        # Remove oldest packet
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put_nowait(packet)
                        except queue.Empty:
                            pass
                            
                # Small sleep to prevent busy waiting
                time.sleep(packet_interval / 10)
                
            except Exception as e:
                self.logger.error(f"Error in mock data generation: {e}")
                break
                
        self.logger.info(f"Mock EEG data generation stopped. Generated {self._packets_generated} packets")
        
    def _should_disconnect(self) -> bool:
        """Check if device should simulate disconnection"""
        if self.profile.disconnect_probability <= 0:
            return False
            
        # Calculate per-call probability from per-second probability
        time_delta = 1.0 / self.sampling_rate
        call_probability = self.profile.disconnect_probability * time_delta
        
        return random.random() < call_probability
        
    def _generate_eeg_packet(self, timestamp: float) -> EEGPacket:
        """Generate realistic EEG data packet"""
        # Time-based signal generation
        t = timestamp - self._start_time
        
        channels = {}
        quality = {}
        
        for i, channel in enumerate(self.channels):
            # Generate realistic EEG-like signals
            signal = self._generate_channel_signal(channel, t, i)
            channels[channel] = signal
            
            # Channel quality based on signal strength and profile
            base_quality = random.uniform(*self.profile.signal_quality_range)
            noise_factor = random.uniform(0.9, 1.1)
            quality[channel] = max(0.0, min(1.0, base_quality * noise_factor))
            
        # Update current signal quality (average of all channels)
        self._current_signal_quality = sum(quality.values()) / len(quality) if quality else 0.0
        
        # Generate power band data
        power_bands = self._generate_power_bands(channels)
        
        return EEGPacket(
            timestamp=timestamp,
            source=EEGSourceType.MOCK,
            channels=channels,
            quality=quality,
            power_bands=power_bands,
            session_id=self.session_id,
            raw_data={
                "mock": True,
                "device_id": self.device_id,
                "packet_count": self._packets_generated,
                "mode": self.mode
            }
        )
        
    def _generate_channel_signal(self, channel: str, t: float, channel_index: int) -> float:
        """Generate realistic signal for a specific channel"""
        # Base frequencies for different brain wave types
        delta = 2.0 * math.sin(2 * math.pi * 2 * t)      # 2 Hz delta
        theta = 1.5 * math.sin(2 * math.pi * 6 * t)      # 6 Hz theta  
        alpha = 3.0 * math.sin(2 * math.pi * 10 * t)     # 10 Hz alpha
        beta = 1.0 * math.sin(2 * math.pi * 20 * t)      # 20 Hz beta
        gamma = 0.5 * math.sin(2 * math.pi * 40 * t)     # 40 Hz gamma
        
        # Channel-specific phase shifts for realism
        phase_shift = channel_index * 0.1
        
        # Combine waves with channel-specific weighting
        if "F" in channel:  # Frontal channels - more beta/gamma
            signal = 0.3*delta + 0.4*theta + 0.6*alpha + 1.2*beta + 0.8*gamma
        elif "O" in channel:  # Occipital channels - more alpha
            signal = 0.2*delta + 0.3*theta + 1.5*alpha + 0.6*beta + 0.3*gamma
        elif "T" in channel:  # Temporal channels - mixed
            signal = 0.4*delta + 0.8*theta + 1.0*alpha + 0.8*beta + 0.4*gamma
        else:  # Other channels
            signal = 0.3*delta + 0.6*theta + 1.0*alpha + 0.7*beta + 0.4*gamma
            
        # Add phase shift
        signal = signal * math.cos(phase_shift)
        
        # Add noise based on profile
        noise = random.gauss(0, self.profile.noise_level)
        signal += noise
        
        # Scale to typical EEG microvolts range
        signal *= 10.0  # Scale to ±30μV range approximately
        
        return signal
        
    def _generate_power_bands(self, channels: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Generate power band data from channel signals"""
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
        power_bands = {}
        
        for channel, signal in channels.items():
            channel_powers = {}
            
            # Simple power estimation (in real system would use FFT)
            for i, band in enumerate(bands):
                # Simulate different power levels for different bands
                base_power = abs(signal) * random.uniform(0.5, 1.5)
                
                # Band-specific scaling
                if band == "alpha":
                    base_power *= 1.5  # Alpha typically dominant
                elif band == "delta":
                    base_power *= 0.8
                elif band == "gamma":
                    base_power *= 0.3
                    
                channel_powers[band] = max(0.1, base_power)
                
            power_bands[channel] = channel_powers
            
        return power_bands
        
    def get_connection_info(self) -> ConnectionInfo:
        """Get detailed connection information"""
        # Calculate connection stability
        stability = "unknown"
        if self._status == ConnectionStatus.DISCONNECTED:
            stability = "disconnected"
        elif self._status in [ConnectionStatus.CONNECTING, ConnectionStatus.ERROR]:
            stability = "unstable"
        elif self._status == ConnectionStatus.CONNECTED:
            stability = "stable"
        elif self._status == ConnectionStatus.STREAMING:
            # Check if data is flowing
            time_since_data = time.time() - self._last_data_time
            if time_since_data < 2.0:
                stability = "stable"
            else:
                stability = "unstable"
                
        device_info = {
            "device_id": self.device_id,
            "model": "Mock EPOC+",
            "firmware": "1.0.0",
            "mode": self.mode,
            "profile": self.profile.name,
            "connection_attempts": self.connection_attempts,
            "packets_generated": self._packets_generated,
            "uptime_s": time.time() - self.connection_start_time if self.connection_start_time else 0
        }
        
        return ConnectionInfo(
            status=self._status,
            source=EEGSourceType.MOCK,
            device_info=device_info,
            signal_quality=self._current_signal_quality,
            last_data_timestamp=self._last_data_time if self._last_data_time > 0 else None,
            connection_stability=stability,
            error_message=self._error_message
        )
        
    def get_latest_packet(self) -> Optional[EEGPacket]:
        """Get the latest EEG data packet"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
            
    async def _async_delay(self, seconds: float):
        """Async delay that doesn't block the event loop"""
        import asyncio
        await asyncio.sleep(seconds)