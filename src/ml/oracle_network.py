"""
Oracle Network Client

This module provides WebSocket client capability for the 369 Oracle
to connect to multiple inference streams and receive real-time data.
"""

import json
import time
import asyncio
import threading
import websockets
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

# Simple prediction result for network communication
@dataclass
class PredictionResult:
    """Result from model prediction (simplified for network)"""
    timestamp: float
    mode: int
    colors: np.ndarray
    curves: np.ndarray
    dials: np.ndarray
    confidence: float
    input_data: Dict[str, Any]


@dataclass
class OracleStreamConnection:
    """Configuration for connecting to an inference stream"""
    stream_id: str
    host: str = "localhost"
    port: int = 8765
    layer: int = 1  # Oracle layer (1=Primary, 2=Subconscious, 3=Universal)
    enabled: bool = True


class OracleStreamClient:
    """WebSocket client for receiving inference stream data"""
    
    def __init__(self, connection: OracleStreamConnection):
        self.connection = connection
        self.logger = logging.getLogger(f"oracle_client_{connection.stream_id}")
        
        # Connection management
        self.websocket = None
        self.loop = None
        self.thread = None
        self.is_connected = False
        self.is_running = False
        
        # Data callbacks
        self.prediction_callbacks: List[Callable[[str, PredictionResult], None]] = []
        self.status_callbacks: List[Callable[[str, str], None]] = []
        
        # Statistics
        self.stats = {
            'connection_attempts': 0,
            'messages_received': 0,
            'errors': 0,
            'last_message_time': None,
            'connect_time': None
        }
        
    async def connect(self):
        """Connect to the inference stream"""
        uri = f"ws://{self.connection.host}:{self.connection.port}"
        self.stats['connection_attempts'] += 1
        
        try:
            self.logger.info(f"Connecting to {uri}...")
            self.websocket = await websockets.connect(uri)
            self.is_connected = True
            self.stats['connect_time'] = time.time()
            
            self.logger.info(f"Connected to {self.connection.stream_id}")
            self._notify_status("connected")
            
            # Listen for messages
            await self._listen_loop()
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Connection closed for {self.connection.stream_id}")
            self.is_connected = False
            self._notify_status("disconnected")
            
        except Exception as e:
            self.logger.error(f"Connection error for {self.connection.stream_id}: {e}")
            self.is_connected = False
            self.stats['errors'] += 1
            self._notify_status(f"error: {e}")
            
    async def _listen_loop(self):
        """Listen for incoming messages"""
        async for message in self.websocket:
            try:
                data = json.loads(message)
                await self._handle_message(data)
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {e}")
                self.stats['errors'] += 1
                
            except Exception as e:
                self.logger.error(f"Message handling error: {e}")
                self.stats['errors'] += 1
                
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming message from inference stream"""
        msg_type = data.get('type', 'unknown')
        self.stats['messages_received'] += 1
        self.stats['last_message_time'] = time.time()
        
        if msg_type == 'config':
            # Stream configuration
            stream_id = data.get('stream_id', 'unknown')
            mode = data.get('mode', 0)
            self.logger.info(f"Received config: {stream_id}, mode {mode}")
            
        elif msg_type == 'prediction':
            # Prediction data - convert back to PredictionResult
            prediction_data = data.get('data', {})
            
            # Convert arrays back from JSON
            prediction = PredictionResult(
                timestamp=prediction_data.get('timestamp', time.time()),
                mode=prediction_data.get('mode', 1),
                colors=np.array(prediction_data.get('colors', [])),
                curves=np.array(prediction_data.get('curves', [])),
                dials=np.array(prediction_data.get('dials', [])),
                confidence=prediction_data.get('confidence', 0.0),
                input_data=prediction_data.get('input_data', {})
            )
            
            # Notify callbacks
            self._notify_prediction(prediction)
            
        elif msg_type == 'pong':
            # Ping response
            pass
            
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")
            
    def _notify_prediction(self, prediction: PredictionResult):
        """Notify prediction callbacks"""
        for callback in self.prediction_callbacks:
            try:
                callback(self.connection.stream_id, prediction)
            except Exception as e:
                self.logger.error(f"Error in prediction callback: {e}")
                
    def _notify_status(self, status: str):
        """Notify status callbacks"""
        for callback in self.status_callbacks:
            try:
                callback(self.connection.stream_id, status)
            except Exception as e:
                self.logger.error(f"Error in status callback: {e}")
                
    async def send_ping(self):
        """Send ping to keep connection alive"""
        if self.is_connected and self.websocket:
            try:
                await self.websocket.send(json.dumps({'type': 'ping'}))
            except Exception as e:
                self.logger.error(f"Error sending ping: {e}")
                
    def start(self):
        """Start the client in a background thread"""
        if self.is_running:
            return
            
        self.is_running = True
        
        def run_client():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            try:
                self.loop.run_until_complete(self.connect())
            except Exception as e:
                self.logger.error(f"Client thread error: {e}")
            finally:
                self.loop.close()
                
        self.thread = threading.Thread(target=run_client, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the client"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.is_connected = False
        
        if self.loop and self.websocket:
            # Schedule disconnect in the event loop
            asyncio.run_coroutine_threadsafe(
                self.websocket.close(),
                self.loop
            )
            
            # Stop the event loop
            self.loop.call_soon_threadsafe(self.loop.stop)
            
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        self.logger.info(f"Client stopped for {self.connection.stream_id}")
        
    def add_prediction_callback(self, callback: Callable[[str, PredictionResult], None]):
        """Add callback for prediction results"""
        self.prediction_callbacks.append(callback)
        
    def add_status_callback(self, callback: Callable[[str, str], None]):
        """Add callback for status updates"""
        self.status_callbacks.append(callback)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics"""
        runtime = 0
        if self.stats['connect_time']:
            runtime = time.time() - self.stats['connect_time']
            
        return {
            'stream_id': self.connection.stream_id,
            'layer': self.connection.layer,
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'connection_attempts': self.stats['connection_attempts'],
            'messages_received': self.stats['messages_received'],
            'errors': self.stats['errors'],
            'last_message_time': self.stats['last_message_time'],
            'runtime_seconds': runtime,
            'host': self.connection.host,
            'port': self.connection.port
        }


class Oracle369NetworkManager:
    """Manages network connections for the 369 Oracle system"""
    
    def __init__(self):
        self.clients: Dict[str, OracleStreamClient] = {}
        self.logger = logging.getLogger("oracle_network_manager")
        
        # Data buffers for each layer
        self.layer_data: Dict[int, List[PredictionResult]] = {
            1: [],  # Primary consciousness
            2: [],  # Subconscious
            3: []   # Universal consciousness
        }
        self.max_buffer_size = 100
        
        # Callbacks for Oracle updates
        self.layer_callbacks: List[Callable[[int, PredictionResult], None]] = []
        self.status_callbacks: List[Callable[[str], None]] = []
        
    def add_stream_connection(self, stream_id: str, host: str, port: int, layer: int):
        """Add a new stream connection"""
        connection = OracleStreamConnection(
            stream_id=stream_id,
            host=host,
            port=port,
            layer=layer
        )
        
        client = OracleStreamClient(connection)
        client.add_prediction_callback(self._on_prediction_received)
        client.add_status_callback(self._on_status_update)
        
        self.clients[stream_id] = client
        self.logger.info(f"Added stream connection: {stream_id} -> Layer {layer}")
        
    def start_all_connections(self):
        """Start all stream connections"""
        for stream_id, client in self.clients.items():
            try:
                client.start()
                self.logger.info(f"Started client for {stream_id}")
            except Exception as e:
                self.logger.error(f"Failed to start client {stream_id}: {e}")
                
    def stop_all_connections(self):
        """Stop all stream connections"""
        for stream_id, client in self.clients.items():
            try:
                client.stop()
                self.logger.info(f"Stopped client for {stream_id}")
            except Exception as e:
                self.logger.error(f"Error stopping client {stream_id}: {e}")
                
    def _on_prediction_received(self, stream_id: str, prediction: PredictionResult):
        """Handle prediction from a stream"""
        # Find the layer for this stream
        client = self.clients.get(stream_id)
        if not client:
            return
            
        layer = client.connection.layer
        
        # Add to layer buffer
        self.layer_data[layer].append(prediction)
        
        # Limit buffer size
        if len(self.layer_data[layer]) > self.max_buffer_size:
            self.layer_data[layer] = self.layer_data[layer][-self.max_buffer_size:]
            
        # Notify callbacks
        for callback in self.layer_callbacks:
            try:
                callback(layer, prediction)
            except Exception as e:
                self.logger.error(f"Error in layer callback: {e}")
                
    def _on_status_update(self, stream_id: str, status: str):
        """Handle status update from a stream"""
        message = f"{stream_id}: {status}"
        self.logger.info(f"Status update: {message}")
        
        for callback in self.status_callbacks:
            try:
                callback(message)
            except Exception as e:
                self.logger.error(f"Error in status callback: {e}")
                
    def get_latest_prediction(self, layer: int) -> Optional[PredictionResult]:
        """Get the latest prediction for a specific layer"""
        if layer in self.layer_data and self.layer_data[layer]:
            return self.layer_data[layer][-1]
        return None
        
    def get_layer_buffer(self, layer: int) -> List[PredictionResult]:
        """Get all predictions for a specific layer"""
        return self.layer_data.get(layer, []).copy()
        
    def add_layer_callback(self, callback: Callable[[int, PredictionResult], None]):
        """Add callback for layer updates"""
        self.layer_callbacks.append(callback)
        
    def add_status_callback(self, callback: Callable[[str], None]):
        """Add callback for status updates"""
        self.status_callbacks.append(callback)
        
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all clients"""
        return {
            stream_id: client.get_statistics()
            for stream_id, client in self.clients.items()
        }
        
    def is_all_connected(self) -> bool:
        """Check if all clients are connected"""
        return all(client.is_connected for client in self.clients.values())
        
    def get_connection_status(self) -> Dict[str, bool]:
        """Get connection status for all streams"""
        return {
            stream_id: client.is_connected
            for stream_id, client in self.clients.items()
        }


def create_oracle_network_manager() -> Oracle369NetworkManager:
    """Create network manager with default Oracle stream connections"""
    manager = Oracle369NetworkManager()
    
    # Add the three Oracle layer connections
    manager.add_stream_connection("oracle_layer_1", "localhost", 8765, layer=1)
    manager.add_stream_connection("oracle_layer_2", "localhost", 8766, layer=2)
    manager.add_stream_connection("oracle_layer_3", "localhost", 8767, layer=3)
    
    return manager


if __name__ == "__main__":
    # Test the Oracle network client
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\nShutting down Oracle network...")
        manager.stop_all_connections()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start Oracle network manager
    manager = create_oracle_network_manager()
    
    def on_layer_update(layer: int, prediction: PredictionResult):
        print(f"Layer {layer} prediction: confidence={prediction.confidence:.3f}, "
              f"colors={len(prediction.colors)}, curves={len(prediction.curves)}")
              
    def on_status(message: str):
        print(f"Status: {message}")
        
    manager.add_layer_callback(on_layer_update)
    manager.add_status_callback(on_status)
    
    manager.start_all_connections()
    
    print("Oracle network client started. Waiting for inference streams...")
    print("Expected connections:")
    for stream_id, client in manager.clients.items():
        conn = client.connection
        print(f"  Layer {conn.layer}: ws://{conn.host}:{conn.port}")
        
    print("\nPress Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
            
            # Print connection status every 10 seconds
            if int(time.time()) % 10 == 0:
                status = manager.get_connection_status()
                connected = sum(status.values())
                total = len(status)
                print(f"\nConnection Status: {connected}/{total} connected")
                
                for stream_id, is_connected in status.items():
                    status_str = "✓" if is_connected else "✗"
                    print(f"  {status_str} {stream_id}")
                    
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all_connections()