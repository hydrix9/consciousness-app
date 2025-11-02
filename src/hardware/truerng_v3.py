"""
TrueRNG V3 Hardware Interface Module

This module handles communication with the ubld.it™ TrueRNG V3 device
for generating true random numbers based on silicon electrical noise.
"""

import serial
import time
import threading
import queue
import logging
from typing import Optional, Callable, List
from dataclasses import dataclass
import numpy as np


@dataclass
class RNGSample:
    """Data structure for a single RNG sample"""
    timestamp: float
    raw_bytes: bytes
    values: List[int]
    normalized: List[float]  # Normalized to 0-1 range


class TrueRNGV3:
    """Interface class for TrueRNG V3 device"""
    
    def __init__(self, device_path: str = "auto", baud_rate: int = 3000000, 
                 buffer_size: int = 1024, timeout: float = 1.0):
        """
        Initialize TrueRNG V3 interface
        
        Args:
            device_path: COM port path or "auto" for auto-detection
            baud_rate: Communication baud rate (default 3Mbps)
            buffer_size: Size of internal buffer
            timeout: Communication timeout in seconds
        """
        self.device_path = device_path
        self.baud_rate = baud_rate
        self.buffer_size = buffer_size
        self.timeout = timeout
        
        self.serial_connection: Optional[serial.Serial] = None
        self.is_connected = False
        self.is_streaming = False
        
        # Threading for continuous data collection
        self.data_thread: Optional[threading.Thread] = None
        self.data_queue = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()
        
        # Callbacks for real-time data processing
        self.data_callbacks: List[Callable[[RNGSample], None]] = []
        
        self.logger = logging.getLogger(__name__)
        
    def detect_device(self) -> Optional[str]:
        """
        Auto-detect TrueRNG V3 device on available COM ports
        
        Returns:
            Device path if found, None otherwise
        """
        import serial.tools.list_ports
        
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            try:
                # TrueRNG V3 USB VID:PID is 04D8:F5FE
                if port.vid == 0x04D8 and port.pid == 0xF5FE:
                    self.logger.info(f"TrueRNG V3 detected on {port.device}")
                    return port.device
            except Exception as e:
                self.logger.debug(f"Error checking port {port.device}: {e}")
                continue
                
        # Fallback: try common ports
        common_ports = ["COM3", "COM4", "COM5", "/dev/ttyACM0", "/dev/ttyUSB0"]
        for port in common_ports:
            if self._test_port(port):
                return port
                
        return None
        
    def _test_port(self, port: str) -> bool:
        """Test if a port contains a TrueRNG device"""
        try:
            test_serial = serial.Serial(port, self.baud_rate, timeout=1.0)
            # Try to read some data
            test_data = test_serial.read(10)
            test_serial.close()
            return len(test_data) > 0
        except Exception:
            return False
            
    def connect(self) -> bool:
        """
        Establish connection to TrueRNG V3 device
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.is_connected:
            return True
            
        # Auto-detect device if needed
        if self.device_path == "auto":
            detected_path = self.detect_device()
            if not detected_path:
                self.logger.error("Could not auto-detect TrueRNG V3 device")
                return False
            self.device_path = detected_path
            
        try:
            self.serial_connection = serial.Serial(
                self.device_path,
                self.baud_rate,
                timeout=self.timeout
            )
            
            # Test connection by reading some data
            test_data = self.serial_connection.read(100)
            if len(test_data) < 10:
                raise Exception("Device not responding with sufficient data")
                
            self.is_connected = True
            self.logger.info(f"Connected to TrueRNG V3 on {self.device_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to TrueRNG V3: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from TrueRNG V3 device"""
        self.stop_streaming()
        
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            
        self.is_connected = False
        self.logger.info("Disconnected from TrueRNG V3")
        
    def read_bytes(self, num_bytes: int) -> Optional[bytes]:
        """
        Read raw bytes from TrueRNG device
        
        Args:
            num_bytes: Number of bytes to read
            
        Returns:
            Raw bytes or None if error
        """
        if not self.is_connected or not self.serial_connection:
            return None
            
        try:
            return self.serial_connection.read(num_bytes)
        except Exception as e:
            self.logger.error(f"Error reading from TrueRNG: {e}")
            return None
            
    def read_sample(self, num_bytes: int = 16) -> Optional[RNGSample]:
        """
        Read and process a single sample from TrueRNG
        
        Args:
            num_bytes: Number of bytes to read for this sample
            
        Returns:
            RNGSample object or None if error
        """
        raw_bytes = self.read_bytes(num_bytes)
        if raw_bytes is None:
            return None
            
        timestamp = time.time()
        values = list(raw_bytes)
        normalized = [v / 255.0 for v in values]  # Normalize to 0-1 range
        
        return RNGSample(
            timestamp=timestamp,
            raw_bytes=raw_bytes,
            values=values,
            normalized=normalized
        )
        
    def start_streaming(self, sample_rate: float = 1000.0, bytes_per_sample: int = 16):
        """
        Start continuous data streaming in background thread
        
        Args:
            sample_rate: Desired sampling rate in Hz
            bytes_per_sample: Bytes per sample
        """
        if self.is_streaming:
            return
            
        if not self.is_connected:
            if not self.connect():
                return
                
        self.stop_event.clear()
        self.is_streaming = True
        
        self.data_thread = threading.Thread(
            target=self._streaming_worker,
            args=(sample_rate, bytes_per_sample),
            daemon=True
        )
        self.data_thread.start()
        
        self.logger.info(f"Started RNG streaming at {sample_rate} Hz")
        
    def stop_streaming(self):
        """Stop continuous data streaming"""
        if not self.is_streaming:
            return
            
        self.stop_event.set()
        self.is_streaming = False
        
        if self.data_thread:
            self.data_thread.join(timeout=2.0)
            
        self.logger.info("Stopped RNG streaming")
        
    def _streaming_worker(self, sample_rate: float, bytes_per_sample: int):
        """Background worker for continuous data collection"""
        sample_interval = 1.0 / sample_rate
        
        while not self.stop_event.is_set():
            start_time = time.time()
            
            sample = self.read_sample(bytes_per_sample)
            if sample:
                # Add to queue (non-blocking)
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
                        self.logger.error(f"Error in RNG callback: {e}")
                        
            # Maintain sample rate
            elapsed = time.time() - start_time
            sleep_time = max(0, sample_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def get_latest_samples(self, max_samples: int = 100) -> List[RNGSample]:
        """
        Get latest samples from the queue
        
        Args:
            max_samples: Maximum number of samples to return
            
        Returns:
            List of recent RNGSample objects
        """
        samples = []
        for _ in range(min(max_samples, self.data_queue.qsize())):
            try:
                samples.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return samples
        
    def add_data_callback(self, callback: Callable[[RNGSample], None]):
        """Add callback function for real-time data processing"""
        self.data_callbacks.append(callback)
        
    def remove_data_callback(self, callback: Callable[[RNGSample], None]):
        """Remove callback function"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
            
    def get_statistics(self, samples: List[RNGSample]) -> dict:
        """
        Calculate statistical properties of RNG samples
        
        Args:
            samples: List of RNG samples
            
        Returns:
            Dictionary with statistical measures
        """
        if not samples:
            return {}
            
        all_values = []
        for sample in samples:
            all_values.extend(sample.normalized)
            
        all_values = np.array(all_values)
        
        return {
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values)),
            "min": float(np.min(all_values)),
            "max": float(np.max(all_values)),
            "entropy": self._calculate_entropy(all_values),
            "sample_count": len(samples),
            "total_bytes": sum(len(s.raw_bytes) for s in samples)
        }
        
    def _calculate_entropy(self, values: np.ndarray) -> float:
        """Calculate Shannon entropy of the data"""
        # Convert to discrete bins for entropy calculation
        bins = np.histogram(values, bins=256, range=(0, 1))[0]
        bins = bins[bins > 0]  # Remove zero counts
        probs = bins / np.sum(bins)
        return float(-np.sum(probs * np.log2(probs)))


# Mock RNG interface for testing without hardware
class MockTrueRNG(TrueRNGV3):
    """Mock TrueRNG interface for testing without hardware"""
    
    def __init__(self, generation_rate_kbps: float = 8.0):
        """
        Initialize mock TrueRNG interface
        
        Args:
            generation_rate_kbps: Data generation rate in kilobits per second
        """
        super().__init__()
        self.generation_rate_kbps = generation_rate_kbps
        self.is_connected = False
        self.is_streaming = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.data_callbacks = []
        self.stop_event = threading.Event()
        self.stream_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Calculate bytes per second (1 kilobit = 125 bytes)
        self.bytes_per_second = generation_rate_kbps * 125
        
    def connect(self) -> bool:
        """Mock connection"""
        self.is_connected = True
        self.logger.info(f"Connected to mock TrueRNG (rate: {self.generation_rate_kbps} kbps)")
        return True
        
    def disconnect(self):
        """Mock disconnection"""
        self.stop_streaming()
        self.is_connected = False
        self.logger.info("Disconnected from mock TrueRNG")
        
    def start_streaming(self, sample_rate: float = 1000.0, bytes_per_sample: int = 16):
        """Start mock data streaming"""
        if self.is_streaming:
            return
            
        # Adjust sample rate based on generation rate
        actual_sample_rate = min(sample_rate, self.bytes_per_second / bytes_per_sample)
        
        self.stop_event.clear()
        self.is_streaming = True
        
        self.stream_thread = threading.Thread(
            target=self._mock_streaming_worker,
            args=(actual_sample_rate, bytes_per_sample),
            daemon=True
        )
        self.stream_thread.start()
        
        self.logger.info(f"Started mock RNG streaming at {actual_sample_rate:.1f} Hz "
                        f"({self.generation_rate_kbps} kbps)")
        
    def _mock_streaming_worker(self, sample_rate: float, bytes_per_sample: int):
        """Generate mock random data"""
        import random
        import struct
        
        sample_interval = 1.0 / sample_rate
        
        while not self.stop_event.is_set():
            start_time = time.time()
            
            # Generate random bytes using Python's random module
            # In a real scenario, you might use os.urandom() or secrets module
            random_bytes = bytes([random.randint(0, 255) for _ in range(bytes_per_sample)])
            
            # Create mock sample
            sample = RNGSample(
                timestamp=start_time,
                raw_bytes=random_bytes,
                values=list(random_bytes),
                normalized=[b / 255.0 for b in random_bytes]
            )
            
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
                    self.logger.error(f"Error in mock RNG callback: {e}")
                    
            # Maintain sample rate
            elapsed = time.time() - start_time
            sleep_time = max(0, sample_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def read_all_samples(self) -> List[RNGSample]:
        """Read all available samples from the queue"""
        samples = []
        while True:
            try:
                sample = self.data_queue.get_nowait()
                samples.append(sample)
            except queue.Empty:
                break
        return samples
    
    def stop_streaming(self):
        """Stop mock data streaming"""
        if not self.is_streaming:
            return
            
        self.stop_event.set()
        self.is_streaming = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
            
        self.logger.info("Stopped mock RNG streaming")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    rng = TrueRNGV3()
    
    if rng.connect():
        print("Connected to TrueRNG V3!")
        
        # Read a few samples
        for i in range(5):
            sample = rng.read_sample()
            if sample:
                print(f"Sample {i+1}: {sample.normalized[:8]}...")
                
        # Test streaming for 2 seconds
        def print_callback(sample: RNGSample):
            print(f"Stream: {sample.normalized[:4]}... at {sample.timestamp}")
            
        rng.add_data_callback(print_callback)
        rng.start_streaming(sample_rate=10.0)
        
        time.sleep(2)
        
        rng.stop_streaming()
        
        # Get statistics
        samples = rng.get_latest_samples()
        stats = rng.get_statistics(samples)
        print(f"Statistics: {stats}")
        
        rng.disconnect()
    else:
        print("Failed to connect to TrueRNG V3")