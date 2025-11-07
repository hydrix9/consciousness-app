"""
EEG Bridge Module

Unified interface for EEG data sources with automatic fallback system.
Inspired by ChronoSword's robust architecture for maximum reliability.
"""

import os
import time
import queue
import threading
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Iterator, Callable
from enum import Enum
import numpy as np


class EEGSourceType(Enum):
    """EEG data source types"""
    CORTEX = "cortex"  # Raw EEG - requires license
    BCI = "bci"  # FREE Emotiv BCI data (Performance Metrics, Mental Commands, Facial Expressions)
    MOCK = "mock"
    SIMULATED = "simulated"
    LSL = "lsl"  # For future Lab Streaming Layer support


class ConnectionStatus(Enum):
    """Connection status states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"
    UNSTABLE = "unstable"


@dataclass
class EEGPacket:
    """Unified EEG data packet structure"""
    timestamp: float
    source: EEGSourceType
    channels: Dict[str, float]  # Channel name -> value mapping
    quality: Dict[str, float]   # Channel quality (0.0-1.0)
    power_bands: Optional[Dict[str, Dict[str, float]]] = None  # Channel -> Band -> Power
    raw_data: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


@dataclass
class ConnectionInfo:
    """Detailed connection information"""
    status: ConnectionStatus
    source: EEGSourceType
    device_info: Dict[str, Any]
    signal_quality: float  # Overall signal quality 0.0-1.0
    last_data_timestamp: Optional[float]
    connection_stability: str  # "stable", "unstable", "lost"
    error_message: Optional[str] = None


class EEGSource(ABC):
    """Abstract base class for EEG data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._status = ConnectionStatus.DISCONNECTED
        self._callbacks: List[Callable[[EEGPacket], None]] = []
        self._error_message: Optional[str] = None
        
    @property
    def status(self) -> ConnectionStatus:
        return self._status
        
    @property
    def error_message(self) -> Optional[str]:
        return self._error_message
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to EEG source"""
        pass
        
    @abstractmethod
    async def disconnect(self):
        """Disconnect from EEG source"""
        pass
        
    @abstractmethod
    async def start_streaming(self) -> bool:
        """Start data streaming"""
        pass
        
    @abstractmethod
    async def stop_streaming(self):
        """Stop data streaming"""
        pass
        
    @abstractmethod
    def get_connection_info(self) -> ConnectionInfo:
        """Get detailed connection information"""
        pass
        
    @abstractmethod
    def get_latest_packet(self) -> Optional[EEGPacket]:
        """Get the latest EEG data packet"""
        pass
        
    def add_callback(self, callback: Callable[[EEGPacket], None]):
        """Add data callback"""
        self._callbacks.append(callback)
        
    def remove_callback(self, callback: Callable[[EEGPacket], None]):
        """Remove data callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            
    def _notify_callbacks(self, packet: EEGPacket):
        """Notify all registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(packet)
            except Exception as e:
                self.logger.error(f"Error in callback: {e}")


class EEGBridge:
    """
    Unified EEG interface with automatic fallback system.
    Tries sources in priority order: Cortex -> Mock -> Simulated
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize EEG bridge
        
        Args:
            config: Configuration dictionary with source preferences and settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Data management
        self.data_queue: "queue.Queue[EEGPacket]" = queue.Queue(maxsize=512)
        self.callbacks: List[Callable[[EEGPacket], None]] = []
        
        # Threading
        self._stop_event = threading.Event()
        self._pump_thread: Optional[threading.Thread] = None
        self._push_period = float(self.config.get("push_period_s", 0.25))  # 4Hz default
        
        # Source management
        self.current_source: Optional[EEGSource] = None
        self.source_priority = self._get_source_priority()
        self.fallback_attempted = False
        
        # Status tracking
        self._last_packet_time = time.time()  # Initialize to current time
        self._connection_stability = "unknown"
        
        self.logger.info(f"EEGBridge initialized with {len(self.source_priority)} sources, push_period={self._push_period}s")
        
    def _get_source_priority(self) -> List[EEGSourceType]:
        """Get source priority order from config"""
        requested_source = self.config.get("source", "auto").lower()
        
        if requested_source == "cortex":
            # Raw EEG (requires license) - fallback to BCI then mock
            return [EEGSourceType.CORTEX, EEGSourceType.BCI, EEGSourceType.MOCK, EEGSourceType.SIMULATED]
        elif requested_source == "bci":
            # FREE BCI data (recommended) - fallback to mock
            return [EEGSourceType.BCI, EEGSourceType.MOCK, EEGSourceType.SIMULATED]
        elif requested_source == "mock":
            return [EEGSourceType.MOCK, EEGSourceType.SIMULATED]
        elif requested_source == "simulated":
            return [EEGSourceType.SIMULATED]
        else:  # auto
            # Default: Try BCI first (free), then cortex (needs license), then mock
            return [EEGSourceType.BCI, EEGSourceType.CORTEX, EEGSourceType.MOCK, EEGSourceType.SIMULATED]
            
    def _create_source(self, source_type: EEGSourceType) -> EEGSource:
        """Factory method to create EEG sources"""
        source_config = self.config.get(source_type.value, {})
        
        if source_type == EEGSourceType.CORTEX:
            from .cortex_websocket import CortexWebSocketSource
            return CortexWebSocketSource(source_config)
        elif source_type == EEGSourceType.BCI:
            from .emotiv_bci import EmotivBCISource
            return EmotivBCISource(self.config)  # BCI needs full config
        elif source_type == EEGSourceType.MOCK:
            from .mock_eeg import EnhancedMockEEG
            return EnhancedMockEEG(source_config)
        elif source_type == EEGSourceType.SIMULATED:
            from .simulated_eeg import SimulatedEEGSource
            return SimulatedEEGSource(source_config)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
            
    async def connect(self) -> bool:
        """
        Connect to EEG source with automatic fallback
        
        Returns:
            True if connection successful (any source), False if all sources failed
        """
        self.logger.info("Starting EEG connection with fallback system...")
        
        for source_type in self.source_priority:
            self.logger.info(f"Attempting connection to {source_type.value}...")
            
            try:
                source = self._create_source(source_type)
                
                if await source.connect():
                    self.current_source = source
                    self.current_source.add_callback(self._on_data_received)
                    
                    # Start streaming immediately after connection
                    if await source.start_streaming():
                        self._start_data_pump()
                        self.logger.info(f"Successfully connected and streaming from {source_type.value}")
                        return True
                    else:
                        self.logger.warning(f"Connected to {source_type.value} but failed to start streaming")
                        await source.disconnect()
                        
            except Exception as e:
                self.logger.warning(f"Failed to connect to {source_type.value}: {e}")
                continue
                
        self.logger.error("All EEG sources failed to connect")
        return False
        
    async def disconnect(self):
        """Disconnect from current EEG source"""
        self._stop_data_pump()
        
        if self.current_source:
            try:
                await self.current_source.stop_streaming()
                await self.current_source.disconnect()
                self.logger.info(f"Disconnected from {self.current_source.__class__.__name__}")
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")
            finally:
                self.current_source = None
                
    def _start_data_pump(self):
        """Start background data pump thread"""
        if self._pump_thread and self._pump_thread.is_alive():
            return
            
        self._stop_event.clear()
        self._pump_thread = threading.Thread(
            target=self._data_pump_worker,
            name="EEGDataPump",
            daemon=True
        )
        self._pump_thread.start()
        self.logger.info("Started EEG data pump")
        
    def _stop_data_pump(self):
        """Stop background data pump thread"""
        self._stop_event.set()
        
        if self._pump_thread and self._pump_thread.is_alive():
            self._pump_thread.join(timeout=2.0)
            if self._pump_thread.is_alive():
                self.logger.warning("Data pump thread did not stop gracefully")
                
    def _data_pump_worker(self):
        """Background worker for data collection and distribution"""
        last_push_time = 0.0
        packet_count = 0
        
        self.logger.info(f"EEG data pump started with push_period={self._push_period}s")
        
        while not self._stop_event.is_set():
            try:
                if not self.current_source:
                    time.sleep(0.1)
                    continue
                    
                packet = self.current_source.get_latest_packet()
                if packet:
                    packet_count += 1
                    now = time.time()
                    
                    # Throttle updates based on push period
                    if now - last_push_time >= self._push_period:
                        # Add to queue
                        try:
                            self.data_queue.put_nowait(packet)
                        except queue.Full:
                            # Remove oldest packet and add new one
                            try:
                                self.data_queue.get_nowait()
                                self.data_queue.put_nowait(packet)
                            except queue.Empty:
                                pass
                                
                        # Notify callbacks
                        self._notify_callbacks(packet)
                        
                        last_push_time = now
                        self._last_packet_time = now
                        
                        # Log status occasionally
                        if packet_count % 100 == 0:
                            self.logger.debug(f"Processed {packet_count} packets, queue size: {self.data_queue.qsize()}")
                            
                else:
                    # Check for connection loss
                    if time.time() - self._last_packet_time > 5.0:  # 5 seconds without data
                        self.logger.warning("No data received for 5 seconds, checking connection...")
                        # Use a separate thread for async operation to avoid event loop issues
                        import threading
                        def check_health():
                            try:
                                import asyncio
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                loop.run_until_complete(self._check_connection_health())
                                loop.close()
                            except Exception as e:
                                self.logger.error(f"Error in health check: {e}")
                        
                        health_thread = threading.Thread(target=check_health, daemon=True)
                        health_thread.start()
                        
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error in data pump: {e}")
                time.sleep(0.1)
                
    async def _check_connection_health(self):
        """Check connection health and attempt recovery if needed"""
        if not self.current_source:
            return
            
        try:
            connection_info = self.current_source.get_connection_info()
            
            if connection_info.status in [ConnectionStatus.ERROR, ConnectionStatus.DISCONNECTED]:
                self.logger.warning("Connection lost, attempting recovery...")
                
                # Try to reconnect current source
                if await self.current_source.connect():
                    if await self.current_source.start_streaming():
                        self.logger.info("Connection recovered successfully")
                        return
                        
                # If recovery failed, try fallback
                self.logger.warning("Recovery failed, attempting fallback...")
                await self._attempt_fallback()
                
        except Exception as e:
            self.logger.error(f"Error checking connection health: {e}")
            
    async def _attempt_fallback(self):
        """Attempt to fallback to next available source"""
        if not self.current_source:
            return
            
        current_source_type = None
        for source_type in EEGSourceType:
            if isinstance(self.current_source, self._get_source_class(source_type)):
                current_source_type = source_type
                break
                
        if current_source_type:
            # Find next source in priority list
            try:
                current_index = self.source_priority.index(current_source_type)
                fallback_sources = self.source_priority[current_index + 1:]
                
                for source_type in fallback_sources:
                    self.logger.info(f"Attempting fallback to {source_type.value}...")
                    
                    try:
                        # Disconnect current source
                        await self.current_source.disconnect()
                        
                        # Try new source
                        new_source = self._create_source(source_type)
                        if await new_source.connect():
                            if await new_source.start_streaming():
                                self.current_source = new_source
                                self.current_source.add_callback(self._on_data_received)
                                self.logger.info(f"Successfully failed over to {source_type.value}")
                                return
                                
                    except Exception as e:
                        self.logger.warning(f"Fallback to {source_type.value} failed: {e}")
                        continue
                        
            except ValueError:
                pass  # Current source not in priority list
                
        self.logger.error("All fallback attempts failed")
        
    def _get_source_class(self, source_type: EEGSourceType):
        """Get the class for a source type"""
        if source_type == EEGSourceType.CORTEX:
            from .cortex_websocket import CortexWebSocketSource
            return CortexWebSocketSource
        elif source_type == EEGSourceType.BCI:
            from .emotiv_bci import EmotivBCISource
            return EmotivBCISource
        elif source_type == EEGSourceType.MOCK:
            from .mock_eeg import EnhancedMockEEG
            return EnhancedMockEEG
        elif source_type == EEGSourceType.SIMULATED:
            from .simulated_eeg import SimulatedEEGSource
            return SimulatedEEGSource
        return None
        
    def _on_data_received(self, packet: EEGPacket):
        """Callback for when data is received from source"""
        self._last_packet_time = time.time()
        
    def _notify_callbacks(self, packet: EEGPacket):
        """Notify all registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(packet)
            except Exception as e:
                self.logger.error(f"Error in callback: {e}")
                
    def get_connection_info(self) -> ConnectionInfo:
        """Get detailed connection information"""
        if not self.current_source:
            return ConnectionInfo(
                status=ConnectionStatus.DISCONNECTED,
                source=EEGSourceType.SIMULATED,  # Default
                device_info={},
                signal_quality=0.0,
                last_data_timestamp=None,
                connection_stability="disconnected"
            )
            
        return self.current_source.get_connection_info()
        
    def get_latest_packet(self) -> Optional[EEGPacket]:
        """Get latest EEG packet from queue"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
            
    def get_latest_packets(self, max_count: int = 10) -> List[EEGPacket]:
        """Get multiple latest packets"""
        packets = []
        for _ in range(min(max_count, self.data_queue.qsize())):
            try:
                packets.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return packets
        
    def add_callback(self, callback: Callable[[EEGPacket], None]):
        """Add data callback"""
        self.callbacks.append(callback)
        
    def add_data_callback(self, callback: Callable[[EEGPacket], None]):
        """Add data callback (alias for add_callback for compatibility)"""
        self.add_callback(callback)
        
    def remove_callback(self, callback: Callable[[EEGPacket], None]):
        """Remove data callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            
    def is_connected(self) -> bool:
        """Check if any source is connected"""
        if not self.current_source:
            return False
        return self.current_source.status in [ConnectionStatus.CONNECTED, ConnectionStatus.STREAMING]
        
    def is_streaming(self) -> bool:
        """Check if currently streaming data"""
        if not self.current_source:
            return False
        return self.current_source.status == ConnectionStatus.STREAMING
        
    def get_source_type(self) -> Optional[EEGSourceType]:
        """Get current source type"""
        if not self.current_source:
            return None
            
        for source_type in EEGSourceType:
            if isinstance(self.current_source, self._get_source_class(source_type)):
                return source_type
        return None


# Configuration helper
def load_eeg_config() -> Dict[str, Any]:
    """Load EEG configuration from environment and files"""
    config = {
        "source": os.getenv("EEG_SOURCE", "auto"),
        "push_period_s": float(os.getenv("EEG_UPDATE_RATE_HZ", "4")) ** -1,  # Convert Hz to period
        "cortex": {
            "url": os.getenv("EMOTIV_CORTEX_URL", "wss://127.0.0.1:6868"),
            "client_id": os.getenv("EMOTIV_CLIENT_ID", ""),
            "client_secret": os.getenv("EMOTIV_CLIENT_SECRET", ""),
            "license_key": os.getenv("EMOTIV_LICENSE_KEY", ""),
            "headset_id": os.getenv("EMOTIV_HEADSET_ID", "AUTO"),
            "streams": ["pow", "eeg"],
            "channels": ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"],
            "timeout_s": 30.0
        },
        "mock": {
            "mode": os.getenv("EEG_MOCK_MODE", "stable"),  # stable, unstable, disconnected
            "connection_delay_s": 1.0,
            "signal_quality": 0.8,
            "disconnect_probability": 0.0
        },
        "simulated": {
            "frequency_bands": ["delta", "theta", "alpha", "beta", "gamma"],
            "base_amplitude": 1.0,
            "noise_level": 0.05
        }
    }
    
    # Try to load from YAML files
    try:
        import yaml
        
        # First try app_config.yaml (main config with Emotiv credentials)
        app_config_file = os.path.join(os.path.dirname(__file__), "..", "..", "config", "app_config.yaml")
        if os.path.exists(app_config_file):
            with open(app_config_file, 'r') as f:
                app_config = yaml.safe_load(f)
                if app_config:
                    # Extract Emotiv credentials from hardware.emotiv section
                    emotiv_config = app_config.get('hardware', {}).get('emotiv', {})
                    if emotiv_config:
                        # Map emotiv config to cortex config
                        if emotiv_config.get('client_id'):
                            config['cortex']['client_id'] = emotiv_config['client_id']
                        if emotiv_config.get('client_secret'):
                            config['cortex']['client_secret'] = emotiv_config['client_secret']
                        if emotiv_config.get('license'):
                            config['cortex']['license_key'] = emotiv_config['license']
                        if emotiv_config.get('headset_id'):
                            config['cortex']['headset_id'] = emotiv_config['headset_id']
                    
                    # Also get sampling rate from timing config
                    sampling_rate = app_config.get('timing', {}).get('eeg_sampling_rate', 128)
                    config['cortex']['sampling_rate'] = sampling_rate
        
        # Then try eeg_config.yaml (specific EEG config that can override)
        eeg_config_file = os.path.join(os.path.dirname(__file__), "..", "..", "config", "eeg_config.yaml")
        if os.path.exists(eeg_config_file):
            with open(eeg_config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Merge configurations (file overrides everything)
                    config.update(file_config)
    except ImportError:
        pass  # YAML not available
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load EEG config file: {e}")
        
    return config