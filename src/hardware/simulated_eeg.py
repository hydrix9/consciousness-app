"""
Simulated EEG Source

Provides a reliable simulated EEG data source that always works.
Used as the final fallback in the EEG bridge system.
"""

import time
import math
import threading
import queue
import logging
from typing import Dict, Any, Optional

from .eeg_bridge import EEGSource, EEGPacket, ConnectionInfo, ConnectionStatus, EEGSourceType


class SimulatedEEGSource(EEGSource):
    """Always-available simulated EEG source for testing and fallback"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration
        self.channels = config.get("channels", [
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", 
            "P8", "T8", "FC6", "F4", "F8", "AF4"
        ])
        self.sampling_rate = config.get("sampling_rate", 128)  # Hz
        self.frequency_bands = config.get("frequency_bands", 
                                         ["delta", "theta", "alpha", "beta", "gamma"])
        self.base_amplitude = config.get("base_amplitude", 1.0)
        self.noise_level = config.get("noise_level", 0.05)
        
        # Data generation
        self.data_queue = queue.Queue(maxsize=128)
        self._start_time = time.time()
        self._packets_generated = 0
        self._last_data_time = 0.0
        
        # Threading
        self._streaming_thread: Optional[threading.Thread] = None
        self._stop_streaming = threading.Event()
        
        # Session info
        self.session_id = f"simulated_{int(time.time())}"
        
        self.logger.info("Simulated EEG source initialized")
        
    async def connect(self) -> bool:
        """Simulated connection - always succeeds immediately"""
        self._status = ConnectionStatus.CONNECTED
        self._error_message = None
        self.session_id = f"simulated_{int(time.time())}"
        
        self.logger.info("Simulated EEG connected (always succeeds)")
        return True
        
    async def disconnect(self):
        """Simulated disconnection"""
        await self.stop_streaming()
        self._status = ConnectionStatus.DISCONNECTED
        self.logger.info("Simulated EEG disconnected")
        
    async def start_streaming(self) -> bool:
        """Start simulated data streaming"""
        if self._streaming_thread and self._streaming_thread.is_alive():
            return True  # Already streaming
            
        self._stop_streaming.clear()
        self._streaming_thread = threading.Thread(
            target=self._data_generation_worker,
            name="SimulatedEEGStream",
            daemon=True
        )
        self._streaming_thread.start()
        
        self._status = ConnectionStatus.STREAMING
        self.logger.info("Simulated EEG streaming started")
        return True
        
    async def stop_streaming(self):
        """Stop simulated data streaming"""
        self._stop_streaming.set()
        
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=1.0)
            
        if self._status == ConnectionStatus.STREAMING:
            self._status = ConnectionStatus.CONNECTED
            
        self.logger.info("Simulated EEG streaming stopped")
        
    def _data_generation_worker(self):
        """Background thread for generating simulated EEG data"""
        self.logger.info(f"Simulated EEG data generation started: {self.sampling_rate}Hz")
        
        packet_interval = 1.0 / self.sampling_rate
        last_packet_time = time.time()
        
        while not self._stop_streaming.is_set():
            try:
                current_time = time.time()
                
                # Generate packet at correct interval
                if current_time - last_packet_time >= packet_interval:
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
                self.logger.error(f"Error in simulated data generation: {e}")
                break
                
        self.logger.info(f"Simulated EEG data generation stopped. Generated {self._packets_generated} packets")
        
    def _generate_eeg_packet(self, timestamp: float) -> EEGPacket:
        """Generate simulated EEG data packet with realistic brain wave patterns"""
        t = timestamp - self._start_time
        
        channels = {}
        quality = {}
        
        for i, channel in enumerate(self.channels):
            # Generate brain wave signal
            signal = self._generate_brain_waves(t, channel, i)
            channels[channel] = signal
            
            # Simulated channels always have good quality
            quality[channel] = 0.9 + 0.1 * math.sin(0.1 * t + i)  # Slight variation
            
        # Generate power band data
        power_bands = self._generate_power_bands(t)
        
        return EEGPacket(
            timestamp=timestamp,
            source=EEGSourceType.SIMULATED,
            channels=channels,
            quality=quality,
            power_bands=power_bands,
            session_id=self.session_id,
            raw_data={
                "simulated": True,
                "packet_count": self._packets_generated,
                "uptime": t
            }
        )
        
    def _generate_brain_waves(self, t: float, channel: str, channel_index: int) -> float:
        """Generate realistic brain wave signals"""
        # Different wave types with realistic frequencies and amplitudes
        delta = 0.8 * self.base_amplitude * math.sin(2 * math.pi * 2 * t)      # 2 Hz
        theta = 0.6 * self.base_amplitude * math.sin(2 * math.pi * 6 * t)      # 6 Hz
        alpha = 1.0 * self.base_amplitude * math.sin(2 * math.pi * 10 * t)     # 10 Hz
        beta = 0.4 * self.base_amplitude * math.sin(2 * math.pi * 20 * t)      # 20 Hz
        gamma = 0.2 * self.base_amplitude * math.sin(2 * math.pi * 40 * t)     # 40 Hz
        
        # Channel-specific characteristics
        if "F" in channel:  # Frontal - more beta activity
            signal = 0.2*delta + 0.3*theta + 0.7*alpha + 1.0*beta + 0.3*gamma
        elif "O" in channel:  # Occipital - strong alpha
            signal = 0.1*delta + 0.2*theta + 1.2*alpha + 0.5*beta + 0.2*gamma
        elif "T" in channel:  # Temporal - mixed activity
            signal = 0.3*delta + 0.6*theta + 0.8*alpha + 0.6*beta + 0.3*gamma
        elif "P" in channel:  # Parietal - moderate activity
            signal = 0.2*delta + 0.4*theta + 0.9*alpha + 0.7*beta + 0.4*gamma
        else:  # Central and other
            signal = 0.25*delta + 0.5*theta + 0.8*alpha + 0.6*beta + 0.35*gamma
            
        # Add phase offset for spatial diversity
        phase_offset = channel_index * 0.2
        signal *= math.cos(phase_offset)
        
        # Add subtle time-varying modulation
        modulation = 1.0 + 0.1 * math.sin(0.05 * t)  # Slow modulation
        signal *= modulation
        
        # Add noise
        noise = self.noise_level * (2 * (hash(str(t + channel_index)) % 1000) / 1000 - 1)
        signal += noise
        
        # Scale to microvolts range
        return signal * 10.0
        
    def _generate_power_bands(self, t: float) -> Dict[str, Dict[str, float]]:
        """Generate power band data for all channels"""
        power_bands = {}
        
        for i, channel in enumerate(self.channels):
            channel_powers = {}
            
            # Base power levels that vary over time
            base_modulation = 1.0 + 0.2 * math.sin(0.02 * t + i * 0.1)
            
            for j, band in enumerate(self.frequency_bands):
                # Different base powers for different bands
                if band == "delta":
                    base_power = 0.8
                elif band == "theta":
                    base_power = 0.6
                elif band == "alpha":
                    base_power = 1.0  # Strongest
                elif band == "beta":
                    base_power = 0.5
                elif band == "gamma":
                    base_power = 0.3
                else:
                    base_power = 0.5
                    
                # Channel-specific modifications
                if "F" in channel and band == "beta":
                    base_power *= 1.5  # Frontal beta
                elif "O" in channel and band == "alpha":
                    base_power *= 1.8  # Occipital alpha
                elif "T" in channel and band == "theta":
                    base_power *= 1.3  # Temporal theta
                    
                # Time variation
                time_factor = 1.0 + 0.15 * math.sin(0.03 * t + j * 0.5)
                
                # Final power value
                power = base_power * base_modulation * time_factor
                channel_powers[band] = max(0.1, power)
                
            power_bands[channel] = channel_powers
            
        return power_bands
        
    def get_connection_info(self) -> ConnectionInfo:
        """Get connection information - always shows as stable"""
        device_info = {
            "device_type": "Simulated EEG",
            "model": "Consciousness App Simulator",
            "version": "1.0",
            "channels": len(self.channels),
            "sampling_rate": self.sampling_rate,
            "packets_generated": self._packets_generated,
            "uptime_s": time.time() - self._start_time
        }
        
        # Calculate signal quality (always good for simulation)
        signal_quality = 0.95 + 0.05 * math.sin(0.1 * time.time())
        
        return ConnectionInfo(
            status=self._status,
            source=EEGSourceType.SIMULATED,
            device_info=device_info,
            signal_quality=signal_quality,
            last_data_timestamp=self._last_data_time if self._last_data_time > 0 else None,
            connection_stability="stable",
            error_message=self._error_message
        )
        
    def get_latest_packet(self) -> Optional[EEGPacket]:
        """Get the latest EEG data packet"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None