"""
Network Interface for Inference Streams

This module provides networking capability for inference mode to send
real-time prediction data to external consumers like the 369 Oracle.
"""

import json
import time
import asyncio
import threading
import websockets
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, asdict
import numpy as np

# Import PredictionResult locally to avoid circular imports


@dataclass
class StreamConfig:
    """Configuration for inference stream"""
    port: int = 8765
    host: str = "localhost"
    stream_id: str = "inference_1"
    mode: int = 1  # 1 for RNG only, 2 for RNG+EEG
    enabled: bool = True


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class InferenceStreamServer:
    """WebSocket server for streaming inference results"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.logger = logging.getLogger(f"inference_stream_{config.stream_id}")
        
        # Connection management
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        self.loop = None
        self.thread = None
        self.is_running = False
        
        # Data management
        self.prediction_queue = asyncio.Queue(maxsize=100)
        self.stats = {
            'connections': 0,
            'messages_sent': 0,
            'errors': 0,
            'start_time': None
        }
        
    async def start_server(self):
        """Start the WebSocket server"""
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.config.host,
                self.config.port
            )
            self.stats['start_time'] = time.time()
            self.is_running = True
            
            self.logger.info(f"Inference stream server started on {self.config.host}:{self.config.port}")
            self.logger.info(f"Stream ID: {self.config.stream_id}, Mode: {self.config.mode}")
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise
            
    async def handle_client(self, websocket, path):
        """Handle a new client connection"""
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.logger.info(f"New client connected: {client_addr}")
        
        self.clients.add(websocket)
        self.stats['connections'] += 1
        
        try:
            # Send initial configuration
            config_msg = {
                'type': 'config',
                'stream_id': self.config.stream_id,
                'mode': self.config.mode,
                'timestamp': time.time()
            }
            await websocket.send(json.dumps(config_msg, cls=NumpyEncoder))
            
            # Keep connection alive and handle client messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client disconnected: {client_addr}")
            
        except Exception as e:
            self.logger.error(f"Error handling client {client_addr}: {e}")
            self.stats['errors'] += 1
            
        finally:
            self.clients.discard(websocket)
            
    async def handle_client_message(self, websocket, data):
        """Handle message from client"""
        msg_type = data.get('type', 'unknown')
        
        if msg_type == 'ping':
            await websocket.send(json.dumps({
                'type': 'pong',
                'timestamp': time.time()
            }))
            
        elif msg_type == 'get_stats':
            stats = self.get_statistics()
            await websocket.send(json.dumps({
                'type': 'stats',
                'data': stats
            }, cls=NumpyEncoder))
            
        else:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Unknown message type: {msg_type}'
            }))
            
    async def broadcast_prediction(self, prediction):
        """Broadcast prediction to all connected clients"""
        if not self.clients:
            return
            
        # Convert prediction to streamable format
        message = {
            'type': 'prediction',
            'stream_id': self.config.stream_id,
            'data': {
                'timestamp': prediction.timestamp,
                'mode': prediction.mode,
                'colors': prediction.colors,
                'curves': prediction.curves,
                'dials': prediction.dials,
                'confidence': prediction.confidence,
                'input_data': {
                    'rng': prediction.input_data.get('rng'),
                    'eeg': prediction.input_data.get('eeg')
                }
            }
        }
        
        # Broadcast to all clients
        disconnected_clients = set()
        
        for client in self.clients.copy():
            try:
                await client.send(json.dumps(message, cls=NumpyEncoder))
                self.stats['messages_sent'] += 1
                
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
                
            except Exception as e:
                self.logger.error(f"Error sending to client: {e}")
                disconnected_clients.add(client)
                self.stats['errors'] += 1
                
        # Clean up disconnected clients
        for client in disconnected_clients:
            self.clients.discard(client)
            
    def start(self):
        """Start the server in a background thread"""
        if self.is_running:
            return
            
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            try:
                self.loop.run_until_complete(self.start_server())
                self.loop.run_forever()
                
            except Exception as e:
                self.logger.error(f"Server thread error: {e}")
                
            finally:
                self.loop.close()
                
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        
        # Wait for server to start
        timeout = 5.0
        start_time = time.time()
        while not self.is_running and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if not self.is_running:
            raise RuntimeError("Failed to start inference stream server")
            
    def stop(self):
        """Stop the server"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.loop and self.server:
            # Schedule server shutdown in the event loop
            asyncio.run_coroutine_threadsafe(
                self.server.close(),
                self.loop
            )
            
            # Stop the event loop
            self.loop.call_soon_threadsafe(self.loop.stop)
            
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        self.logger.info("Inference stream server stopped")
        
    def send_prediction(self, prediction):
        """Send prediction to connected clients (thread-safe)"""
        if not self.is_running or not self.loop:
            return
            
        try:
            asyncio.run_coroutine_threadsafe(
                self.broadcast_prediction(prediction),
                self.loop
            )
        except Exception as e:
            self.logger.error(f"Error queuing prediction: {e}")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics"""
        runtime = 0
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            
        return {
            'stream_id': self.config.stream_id,
            'mode': self.config.mode,
            'is_running': self.is_running,
            'connected_clients': len(self.clients),
            'total_connections': self.stats['connections'],
            'messages_sent': self.stats['messages_sent'],
            'errors': self.stats['errors'],
            'runtime_seconds': runtime,
            'host': self.config.host,
            'port': self.config.port
        }


class MultiStreamManager:
    """Manages multiple inference streams for the 369 Oracle"""
    
    def __init__(self, base_port: int = 8765):
        self.base_port = base_port
        self.streams: Dict[str, InferenceStreamServer] = {}
        self.logger = logging.getLogger("multi_stream_manager")
        
    def create_stream(self, stream_id: str, mode: int, port_offset: int = 0) -> InferenceStreamServer:
        """Create a new inference stream"""
        config = StreamConfig(
            port=self.base_port + port_offset,
            stream_id=stream_id,
            mode=mode
        )
        
        stream = InferenceStreamServer(config)
        self.streams[stream_id] = stream
        
        self.logger.info(f"Created stream {stream_id} on port {config.port}")
        return stream
        
    def start_all_streams(self):
        """Start all configured streams"""
        for stream_id, stream in self.streams.items():
            try:
                stream.start()
                self.logger.info(f"Started stream {stream_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to start stream {stream_id}: {e}")
                
    def stop_all_streams(self):
        """Stop all streams"""
        for stream_id, stream in self.streams.items():
            try:
                stream.stop()
                self.logger.info(f"Stopped stream {stream_id}")
                
            except Exception as e:
                self.logger.error(f"Error stopping stream {stream_id}: {e}")
                
    def get_stream(self, stream_id: str) -> Optional[InferenceStreamServer]:
        """Get a specific stream"""
        return self.streams.get(stream_id)
        
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all streams"""
        return {
            stream_id: stream.get_statistics()
            for stream_id, stream in self.streams.items()
        }


# Default configurations for 369 Oracle
ORACLE_STREAM_CONFIGS = [
    StreamConfig(port=8765, stream_id="oracle_layer_1", mode=1),  # Primary consciousness
    StreamConfig(port=8766, stream_id="oracle_layer_2", mode=2),  # Subconscious 
    StreamConfig(port=8767, stream_id="oracle_layer_3", mode=2),  # Universal consciousness
]


def create_oracle_streams() -> MultiStreamManager:
    """Create the three streams for the 369 Oracle system"""
    manager = MultiStreamManager(base_port=8765)
    
    # Create three streams for the Oracle layers
    manager.create_stream("oracle_layer_1", mode=1, port_offset=0)  # Port 8765
    manager.create_stream("oracle_layer_2", mode=2, port_offset=1)  # Port 8766  
    manager.create_stream("oracle_layer_3", mode=2, port_offset=2)  # Port 8767
    
    return manager


if __name__ == "__main__":
    # Test the streaming system
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\nShutting down streams...")
        manager.stop_all_streams()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start streams
    manager = create_oracle_streams()
    manager.start_all_streams()
    
    print("Inference streams started:")
    for stream_id, stream in manager.streams.items():
        config = stream.config
        print(f"  {stream_id}: ws://{config.host}:{config.port} (Mode {config.mode})")
        
    print("\nPress Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
            stats = manager.get_all_statistics()
            
            # Print stats every 10 seconds
            if int(time.time()) % 10 == 0:
                print(f"\nStream Statistics at {time.strftime('%H:%M:%S')}:")
                for stream_id, stat in stats.items():
                    print(f"  {stream_id}: {stat['connected_clients']} clients, "
                          f"{stat['messages_sent']} messages")
                          
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all_streams()