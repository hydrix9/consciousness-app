"""
WebSocket-based Cortex Client

Implements JSON-RPC communication with Emotiv Cortex service via WebSocket.
Provides more reliable connection handling than the direct SDK approach.
Based on ChronoSword's robust implementation.
"""

import json
import time
import asyncio
import threading
import queue
import logging
from typing import Dict, Any, Optional, List
import uuid

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets library not available. Install with: pip install websockets")

from .eeg_bridge import EEGSource, EEGPacket, ConnectionInfo, ConnectionStatus, EEGSourceType


class CortexError(Exception):
    """Cortex-specific error"""
    pass


class CortexWebSocketClient:
    """WebSocket client for Emotiv Cortex JSON-RPC API"""
    
    def __init__(self, url: str, client_id: str, client_secret: str, 
                 license_key: str, headset_id: str = "AUTO"):
        self.url = url
        self.client_id = client_id
        self.client_secret = client_secret
        self.license_key = license_key
        self.headset_id = headset_id
        
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.auth_token: Optional[str] = None
        self.session_id: Optional[str] = None
        
        # Request/response handling
        self.request_id = 0
        self.pending_requests: Dict[int, asyncio.Future] = {}
        
        # Event handling
        self.event_callbacks: Dict[str, callable] = {}
        
        # Connection state
        self.is_connected = False
        self.is_authenticated = False
        
        self.logger = logging.getLogger(__name__)
        
    async def connect(self, timeout: float = 10.0) -> bool:
        """Connect to Cortex WebSocket service"""
        if not WEBSOCKETS_AVAILABLE:
            raise CortexError("websockets library not available")
            
        try:
            self.logger.info(f"Connecting to Cortex at {self.url}")
            
            # Prepare SSL context for localhost connections
            ssl_context = None
            if "127.0.0.1" in self.url or "localhost" in self.url:
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            # Connect with timeout
            self.websocket = await asyncio.wait_for(
                websockets.connect(self.url, ssl=ssl_context),
                timeout=timeout
            )
            
            self.is_connected = True
            self.logger.info("WebSocket connection established")
            
            # Start message handler
            asyncio.create_task(self._message_handler())
            
            return True
            
        except asyncio.TimeoutError:
            self.logger.error(f"Connection timeout after {timeout}s")
            raise CortexError(f"Connection timeout")
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            raise CortexError(f"Connection failed: {e}")
            
    async def disconnect(self):
        """Disconnect from Cortex service"""
        self.is_connected = False
        self.is_authenticated = False
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                self.logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None
                
        # Cancel pending requests
        for future in self.pending_requests.values():
            if not future.done():
                future.cancel()
        self.pending_requests.clear()
        
        self.logger.info("Disconnected from Cortex")
        
    async def _message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON received: {e}")
                except Exception as e:
                    self.logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"Message handler error: {e}")
            self.is_connected = False
            
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle individual message"""
        # Check if it's a response to a request
        if "id" in data:
            request_id = data["id"]
            if request_id in self.pending_requests:
                future = self.pending_requests.pop(request_id)
                if not future.cancelled():
                    if "error" in data:
                        future.set_exception(CortexError(data["error"]))
                    else:
                        future.set_result(data.get("result"))
                return
                
        # Check if it's an event/stream data
        method = data.get("method")
        if method and method in self.event_callbacks:
            try:
                self.event_callbacks[method](data.get("params", {}))
            except Exception as e:
                self.logger.error(f"Error in event callback for {method}: {e}")
                
    async def _send_request(self, method: str, params: Dict[str, Any] = None, 
                           timeout: float = 10.0) -> Any:
        """Send JSON-RPC request and wait for response"""
        if not self.is_connected or not self.websocket:
            raise CortexError("Not connected to Cortex")
            
        self.request_id += 1
        request_id = self.request_id
        
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": request_id
        }
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        try:
            # Send request
            await self.websocket.send(json.dumps(request))
            
            # Wait for response
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
            
        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            raise CortexError(f"Request timeout for {method}")
        except Exception as e:
            self.pending_requests.pop(request_id, None)
            raise CortexError(f"Request failed for {method}: {e}")
            
    async def authenticate(self) -> str:
        """Authenticate with Cortex and get auth token"""
        self.logger.info("Authenticating with Cortex...")
        
        # Check if already authenticated
        if self.auth_token and self.is_authenticated:
            return self.auth_token
            
        try:
            # Request authentication
            result = await self._send_request("authorize", {
                "clientId": self.client_id,
                "clientSecret": self.client_secret,
                "license": self.license_key
            })
            
            self.auth_token = result.get("cortexToken")
            if not self.auth_token:
                raise CortexError("No auth token received")
                
            self.is_authenticated = True
            self.logger.info("Successfully authenticated with Cortex")
            return self.auth_token
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            raise CortexError(f"Authentication failed: {e}")
            
    async def query_headsets(self) -> List[Dict[str, Any]]:
        """Query available headsets"""
        try:
            result = await self._send_request("queryHeadsets")
            return result if isinstance(result, list) else []
        except Exception as e:
            self.logger.error(f"Failed to query headsets: {e}")
            return []
            
    async def create_session(self) -> str:
        """Create a new session"""
        if not self.auth_token:
            await self.authenticate()
            
        # Get available headsets
        headsets = await self.query_headsets()
        connected_headsets = [h for h in headsets if h.get("status") == "connected"]
        
        if not connected_headsets:
            raise CortexError("No connected headsets found")
            
        # Select headset
        if self.headset_id == "AUTO":
            selected_headset = connected_headsets[0]["id"]
        else:
            selected_headset = self.headset_id
            
        self.logger.info(f"Creating session with headset: {selected_headset}")
        
        try:
            result = await self._send_request("createSession", {
                "cortexToken": self.auth_token,
                "headset": selected_headset,
                "status": "active"
            })
            
            self.session_id = result.get("id")
            if not self.session_id:
                raise CortexError("No session ID received")
                
            self.logger.info(f"Session created: {self.session_id}")
            return self.session_id
            
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            raise CortexError(f"Session creation failed: {e}")
            
    async def subscribe_stream(self, streams: List[str]):
        """Subscribe to data streams"""
        if not self.session_id:
            await self.create_session()
            
        try:
            await self._send_request("subscribe", {
                "cortexToken": self.auth_token,
                "session": self.session_id,
                "streams": streams
            })
            
            self.logger.info(f"Subscribed to streams: {streams}")
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to streams: {e}")
            raise CortexError(f"Stream subscription failed: {e}")
            
    async def unsubscribe_stream(self, streams: List[str]):
        """Unsubscribe from data streams"""
        if not self.session_id:
            return
            
        try:
            await self._send_request("unsubscribe", {
                "cortexToken": self.auth_token,
                "session": self.session_id,
                "streams": streams
            })
            
            self.logger.info(f"Unsubscribed from streams: {streams}")
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from streams: {e}")
            
    async def close_session(self):
        """Close the current session"""
        if not self.session_id:
            return
            
        try:
            await self._send_request("updateSession", {
                "cortexToken": self.auth_token,
                "session": self.session_id,
                "status": "close"
            })
            
            self.logger.info("Session closed")
            self.session_id = None
            
        except Exception as e:
            self.logger.error(f"Failed to close session: {e}")
            
    def on_stream_data(self, callback: callable):
        """Register callback for stream data"""
        self.event_callbacks["eeg"] = callback
        self.event_callbacks["pow"] = callback
        self.event_callbacks["met"] = callback
        
    def remove_stream_callback(self):
        """Remove stream data callbacks"""
        self.event_callbacks.pop("eeg", None)
        self.event_callbacks.pop("pow", None)
        self.event_callbacks.pop("met", None)


class CortexWebSocketSource(EEGSource):
    """EEG source using WebSocket Cortex client"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Cortex configuration
        self.url = config.get("url", "wss://127.0.0.1:6868")
        self.client_id = config.get("client_id", "")
        self.client_secret = config.get("client_secret", "")
        self.license_key = config.get("license_key", "")
        self.headset_id = config.get("headset_id", "AUTO")
        self.streams = config.get("streams", ["pow", "eeg"])
        self.channels = config.get("channels", [
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", 
            "P8", "T8", "FC6", "F4", "F8", "AF4"
        ])
        self.timeout = config.get("timeout_s", 30.0)
        
        # Client and data
        self.client: Optional[CortexWebSocketClient] = None
        self.data_queue = queue.Queue(maxsize=256)
        self._last_data_time = 0.0
        self._packets_received = 0
        
        # Device info
        self.device_info = {}
        self.signal_quality = 0.0
        
        self.logger.info("Cortex WebSocket source initialized")
        
    async def connect(self) -> bool:
        """Connect to Cortex via WebSocket"""
        try:
            self._status = ConnectionStatus.CONNECTING
            self._error_message = None
            
            # Validate configuration
            if not all([self.client_id, self.client_secret]):
                self._error_message = "Missing Cortex credentials (client_id, client_secret)"
                self._status = ConnectionStatus.ERROR
                self.logger.error(self._error_message)
                return False
                
            # Create and connect client
            self.client = CortexWebSocketClient(
                self.url, self.client_id, self.client_secret, 
                self.license_key, self.headset_id
            )
            
            await self.client.connect(timeout=self.timeout)
            await self.client.authenticate()
            
            # Get device info
            headsets = await self.client.query_headsets()
            connected_headsets = [h for h in headsets if h.get("status") == "connected"]
            
            if connected_headsets:
                self.device_info = connected_headsets[0]
                self.logger.info(f"Found connected headset: {self.device_info}")
            else:
                self._error_message = "No connected Emotiv headsets found"
                self._status = ConnectionStatus.ERROR
                self.logger.error(self._error_message)
                return False
                
            # Create session
            await self.client.create_session()
            
            self._status = ConnectionStatus.CONNECTED
            self.logger.info("Successfully connected to Cortex via WebSocket")
            return True
            
        except Exception as e:
            self._error_message = f"Cortex connection failed: {e}"
            self._status = ConnectionStatus.ERROR
            self.logger.error(self._error_message)
            return False
            
    async def disconnect(self):
        """Disconnect from Cortex"""
        await self.stop_streaming()
        
        if self.client:
            try:
                await self.client.close_session()
                await self.client.disconnect()
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")
            finally:
                self.client = None
                
        self._status = ConnectionStatus.DISCONNECTED
        self.logger.info("Disconnected from Cortex")
        
    async def start_streaming(self) -> bool:
        """Start EEG data streaming"""
        if not self.client or self._status != ConnectionStatus.CONNECTED:
            return False
            
        try:
            # Set up data callback
            self.client.on_stream_data(self._on_stream_data)
            
            # Subscribe to streams
            await self.client.subscribe_stream(self.streams)
            
            self._status = ConnectionStatus.STREAMING
            self.logger.info(f"Started streaming: {self.streams}")
            return True
            
        except Exception as e:
            self._error_message = f"Failed to start streaming: {e}"
            self.logger.error(self._error_message)
            return False
            
    async def stop_streaming(self):
        """Stop EEG data streaming"""
        if not self.client:
            return
            
        try:
            self.client.remove_stream_callback()
            await self.client.unsubscribe_stream(self.streams)
            
            if self._status == ConnectionStatus.STREAMING:
                self._status = ConnectionStatus.CONNECTED
                
            self.logger.info("Stopped streaming")
            
        except Exception as e:
            self.logger.error(f"Error stopping streaming: {e}")
            
    def _on_stream_data(self, data: Dict[str, Any]):
        """Handle incoming stream data"""
        try:
            stream_name = data.get("streamName", "")
            stream_data = data.get("data", [])
            timestamp = data.get("time", time.time())
            
            if stream_name in ["eeg", "pow"] and stream_data:
                packet = self._create_eeg_packet(stream_name, stream_data, timestamp)
                
                # Add to queue
                try:
                    self.data_queue.put_nowait(packet)
                    self._notify_callbacks(packet)
                    
                    self._packets_received += 1
                    self._last_data_time = time.time()
                    
                except queue.Full:
                    # Remove oldest packet
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put_nowait(packet)
                    except queue.Empty:
                        pass
                        
        except Exception as e:
            self.logger.error(f"Error processing stream data: {e}")
            
    def _create_eeg_packet(self, stream_name: str, stream_data: List[float], 
                          timestamp: float) -> EEGPacket:
        """Create EEG packet from stream data"""
        channels = {}
        quality = {}
        power_bands = None
        
        if stream_name == "eeg":
            # EEG data: [timestamp, counter, channel1, channel2, ...]
            if len(stream_data) > 2:
                for i, value in enumerate(stream_data[2:]):
                    if i < len(self.channels):
                        channels[self.channels[i]] = float(value)
                        quality[self.channels[i]] = 0.8  # Default quality
                        
        elif stream_name == "pow":
            # Power band data: more complex structure
            # This would need to be parsed based on actual Cortex pow format
            if isinstance(stream_data, list) and len(stream_data) > 0:
                # Simplified power band parsing
                bands = ["delta", "theta", "alpha", "beta", "gamma"]
                power_bands = {}
                
                for i, channel in enumerate(self.channels):
                    if i * 5 + 4 < len(stream_data):
                        channel_powers = {}
                        for j, band in enumerate(bands):
                            idx = i * 5 + j
                            if idx < len(stream_data):
                                channel_powers[band] = float(stream_data[idx])
                        power_bands[channel] = channel_powers
                        
                    channels[channel] = sum(channel_powers.values()) if channel_powers else 0.0
                    quality[channel] = 0.8
                    
        # Update signal quality
        if quality:
            self.signal_quality = sum(quality.values()) / len(quality)
            
        return EEGPacket(
            timestamp=timestamp,
            source=EEGSourceType.CORTEX,
            channels=channels,
            quality=quality,
            power_bands=power_bands,
            session_id=self.client.session_id if self.client else None,
            raw_data={
                "stream_name": stream_name,
                "raw_data": stream_data,
                "packet_count": self._packets_received
            }
        )
        
    def get_connection_info(self) -> ConnectionInfo:
        """Get detailed connection information"""
        stability = "unknown"
        if self._status == ConnectionStatus.DISCONNECTED:
            stability = "disconnected"
        elif self._status in [ConnectionStatus.CONNECTING, ConnectionStatus.ERROR]:
            stability = "unstable"
        elif self._status == ConnectionStatus.STREAMING:
            # Check data flow
            time_since_data = time.time() - self._last_data_time
            stability = "stable" if time_since_data < 3.0 else "unstable"
        elif self._status == ConnectionStatus.CONNECTED:
            stability = "stable"
            
        return ConnectionInfo(
            status=self._status,
            source=EEGSourceType.CORTEX,
            device_info=self.device_info,
            signal_quality=self.signal_quality,
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