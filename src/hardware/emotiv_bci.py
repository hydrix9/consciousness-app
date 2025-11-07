"""
Emotiv BCI Data Source
======================

Uses FREE Emotiv BCI data (Performance Metrics, Mental Commands, Facial Expressions)
instead of raw EEG which requires a paid license.

Available FREE Data:
- Performance Metrics: focus, stress, engagement, excitement, interest, relaxation
- Mental Commands: push, pull, lift, drop, left, right, etc. (if trained)
- Facial Expressions: smile, laugh, clench, surprise, frown, etc.

This replaces the raw 14-channel EEG with computed brain metrics.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
import numpy as np

from .eeg_bridge import EEGSource, EEGPacket, EEGSourceType, ConnectionStatus
from .cortex_websocket import CortexWebSocketClient, CortexError


class EmotivBCISource(EEGSource):
    """
    Emotiv BCI data source using free Performance Metrics.
    
    This source uses Emotiv's computed brain metrics instead of raw EEG:
    - No license required!
    - Still captures brain state data
    - Performance metrics: focus, stress, engagement, etc.
    - Mental commands (if trained in Emotiv software)
    - Facial expressions
    
    The data is transformed into a virtual "EEG" format with channels
    representing different mental states.
    """
    
    # Map BCI metrics to virtual EEG channels
    METRIC_CHANNELS = {
        # Performance Metrics (free)
        'focus': 'PM_FOCUS',
        'stress': 'PM_STRESS', 
        'engagement': 'PM_ENGAGE',
        'excitement': 'PM_EXCITE',
        'interest': 'PM_INTEREST',
        'relaxation': 'PM_RELAX',
        
        # Mental Commands (free if trained)
        'push': 'MC_PUSH',
        'pull': 'MC_PULL',
        'lift': 'MC_LIFT',
        'drop': 'MC_DROP',
        'left': 'MC_LEFT',
        'right': 'MC_RIGHT',
        'rotate_left': 'MC_ROTL',
        'rotate_right': 'MC_ROTR',
        
        # Facial Expressions (free)
        'smile': 'FE_SMILE',
        'clench': 'FE_CLENCH',
        'smirk_left': 'FE_SMIRKL',
        'smirk_right': 'FE_SMIRKR',
        'blink': 'FE_BLINK',
        'wink_left': 'FE_WINKL',
        'wink_right': 'FE_WINKR',
        'surprise': 'FE_SURPRISE',
        'frown': 'FE_FROWN',
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Emotiv BCI source.
        
        Args:
            config: Configuration dict with cortex settings
                {
                    'cortex': {
                        'client_id': str,
                        'client_secret': str,
                        'url': str (optional),
                        'headset_id': str (optional)
                    }
                }
        """
        super().__init__(config)
        
        cortex_config = config.get('cortex', {})
        self.client_id = cortex_config.get('client_id', '')
        self.client_secret = cortex_config.get('client_secret', '')
        self.url = cortex_config.get('url', 'wss://127.0.0.1:6868')
        self.headset_id = cortex_config.get('headset_id', 'AUTO')
        
        # BCI-specific: No license needed!
        self.license_key = ""  # Not required for BCI data
        
        self.cortex_client: Optional[CortexWebSocketClient] = None
        self.session_id: Optional[str] = None
        self.data_queue = asyncio.Queue()
        self.streaming_task: Optional[asyncio.Task] = None
        
        self.logger.info("Emotiv BCI Source initialized (FREE - no license required)")
        
    async def connect(self) -> bool:
        """Connect to Emotiv Cortex and initialize BCI streams"""
        try:
            self._status = ConnectionStatus.CONNECTING
            self.logger.info("Connecting to Emotiv Cortex for BCI data...")
            
            # Create Cortex client
            self.cortex_client = CortexWebSocketClient(
                client_id=self.client_id,
                client_secret=self.client_secret,
                license_key=self.license_key,  # Empty - not needed for BCI
                url=self.url
            )
            
            # Connect WebSocket
            await self.cortex_client.connect()
            
            # Authenticate (works without license for BCI data)
            auth_token = await self.cortex_client.authenticate()
            self.logger.info(f"✅ Authenticated with Cortex (BCI mode - no license)")
            
            # Get headset
            headsets = await self.cortex_client.query_headsets()
            if not headsets:
                raise CortexError("No headset found. Please connect your Emotiv device.")
            
            headset = headsets[0]
            headset_id = headset['id']
            self.logger.info(f"✅ Found headset: {headset_id}")
            
            # Create session
            self.session_id = await self.cortex_client.create_session(headset_id)
            self.logger.info(f"✅ Session created: {self.session_id}")
            
            self._status = ConnectionStatus.CONNECTED
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            self._status = ConnectionStatus.ERROR
            self._error_message = str(e)
            return False
    
    async def disconnect(self):
        """Disconnect from Emotiv Cortex"""
        try:
            self._status = ConnectionStatus.DISCONNECTED
            
            if self.streaming_task:
                self.streaming_task.cancel()
                try:
                    await self.streaming_task
                except asyncio.CancelledError:
                    pass
            
            if self.cortex_client and self.session_id:
                await self.cortex_client.close_session(self.session_id)
                self.session_id = None
            
            if self.cortex_client:
                await self.cortex_client.disconnect()
                self.cortex_client = None
                
            self.logger.info("Disconnected from Emotiv BCI")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
    
    async def start_streaming(self) -> bool:
        """Start streaming BCI data (Performance Metrics, Mental Commands, Facial Expressions)"""
        try:
            if not self.cortex_client or not self.session_id:
                raise CortexError("Not connected. Call connect() first.")
            
            # Subscribe to FREE data streams
            streams_to_subscribe = ['met', 'com', 'fac']  # Performance Metrics, Mental Commands, Facial Expressions
            
            self.logger.info(f"Subscribing to BCI streams: {streams_to_subscribe}")
            
            for stream in streams_to_subscribe:
                try:
                    await self.cortex_client.subscribe(
                        session_id=self.session_id,
                        streams=[stream]
                    )
                    self.logger.info(f"✅ Subscribed to {stream} stream")
                except Exception as e:
                    self.logger.warning(f"Could not subscribe to {stream}: {e}")
            
            # Set up data callback
            self.cortex_client.set_data_callback(self._handle_bci_data)
            
            self._status = ConnectionStatus.STREAMING
            self.logger.info("✅ BCI streaming started (FREE - no license required)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            self._status = ConnectionStatus.ERROR
            self._error_message = str(e)
            return False
    
    async def stop_streaming(self):
        """Stop BCI data streaming"""
        try:
            if self.cortex_client and self.session_id:
                streams = ['met', 'com', 'fac']
                for stream in streams:
                    try:
                        await self.cortex_client.unsubscribe(
                            session_id=self.session_id,
                            streams=[stream]
                        )
                    except Exception as e:
                        self.logger.warning(f"Error unsubscribing from {stream}: {e}")
                
            self._status = ConnectionStatus.CONNECTED
            self.logger.info("BCI streaming stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping streaming: {e}")
    
    def _handle_bci_data(self, data: Dict[str, Any]):
        """
        Handle incoming BCI data and convert to EEG packet format.
        
        Transforms Performance Metrics, Mental Commands, and Facial Expressions
        into virtual EEG channels.
        """
        try:
            # Determine data type
            if not isinstance(data, dict):
                return
            
            # Extract stream type
            stream_type = None
            if 'met' in data:  # Performance Metrics
                stream_type = 'met'
                raw_data = data['met']
            elif 'com' in data:  # Mental Commands
                stream_type = 'com'
                raw_data = data['com']
            elif 'fac' in data:  # Facial Expressions
                stream_type = 'fac'
                raw_data = data['fac']
            else:
                return
            
            # Convert to virtual EEG channels
            channels = self._convert_bci_to_channels(stream_type, raw_data)
            
            # Create EEG packet
            packet = EEGPacket(
                timestamp=time.time(),
                source=EEGSourceType.CORTEX,
                channels=channels,
                quality={ch: 1.0 for ch in channels},  # BCI data is always high quality
                raw_data={'type': stream_type, 'data': raw_data}
            )
            
            # Put in queue for async retrieval
            try:
                self.data_queue.put_nowait(packet)
            except asyncio.QueueFull:
                # Drop oldest packet if queue is full
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put_nowait(packet)
                except:
                    pass
            
            # Call registered callbacks
            for callback in self._callbacks:
                try:
                    callback(packet)
                except Exception as e:
                    self.logger.error(f"Error in callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error handling BCI data: {e}")
    
    def _convert_bci_to_channels(self, stream_type: str, raw_data: List[Any]) -> Dict[str, float]:
        """
        Convert BCI data to virtual EEG channel format.
        
        Args:
            stream_type: 'met', 'com', or 'fac'
            raw_data: Raw data array from Cortex
            
        Returns:
            Dict mapping channel names to values (normalized 0-1)
        """
        channels = {}
        
        try:
            if stream_type == 'met':  # Performance Metrics
                # Format: [timestamp, focus, stress, engagement, excitement, interest, relaxation, ...]
                if len(raw_data) >= 7:
                    channels['PM_FOCUS'] = self._normalize_metric(raw_data[1])
                    channels['PM_STRESS'] = self._normalize_metric(raw_data[2])
                    channels['PM_ENGAGE'] = self._normalize_metric(raw_data[3])
                    channels['PM_EXCITE'] = self._normalize_metric(raw_data[4])
                    channels['PM_INTEREST'] = self._normalize_metric(raw_data[5])
                    channels['PM_RELAX'] = self._normalize_metric(raw_data[6])
                    
            elif stream_type == 'com':  # Mental Commands
                # Format: [timestamp, action, power]
                if len(raw_data) >= 3:
                    action = raw_data[1]
                    power = self._normalize_metric(raw_data[2])
                    
                    # Map action to channel
                    action_channel = self.METRIC_CHANNELS.get(action, f'MC_{action.upper()}')
                    channels[action_channel] = power
                    
            elif stream_type == 'fac':  # Facial Expressions
                # Format: [timestamp, action, power]
                if len(raw_data) >= 3:
                    action = raw_data[1]
                    power = self._normalize_metric(raw_data[2])
                    
                    # Map action to channel
                    action_channel = self.METRIC_CHANNELS.get(action, f'FE_{action.upper()}')
                    channels[action_channel] = power
                    
        except Exception as e:
            self.logger.error(f"Error converting BCI data: {e}")
        
        return channels
    
    def _normalize_metric(self, value: Any) -> float:
        """Normalize BCI metric to 0-1 range"""
        try:
            val = float(value)
            # Most Emotiv metrics are already 0-1, but clamp just in case
            return max(0.0, min(1.0, val))
        except:
            return 0.0
    
    async def get_next_packet(self, timeout: float = 1.0) -> Optional[EEGPacket]:
        """Get next BCI data packet"""
        try:
            return await asyncio.wait_for(self.data_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    def register_callback(self, callback: Any):
        """Register a callback for BCI data packets"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Any):
        """Unregister a callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get BCI connection information"""
        return {
            'status': self._status.value,
            'source': 'emotiv_bci',
            'mode': 'FREE - Performance Metrics + Mental Commands + Facial Expressions',
            'license_required': False,
            'device_info': {
                'session_id': self.session_id,
                'streams': ['Performance Metrics', 'Mental Commands', 'Facial Expressions']
            }
        }
