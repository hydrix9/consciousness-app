#!/usr/bin/env python3
"""
369 Oracle Multi-Inference Launcher

This script launches three inference mode instances on different ports
to feed data to the 369 Oracle system.
"""

import os
import sys
import time
import signal
import subprocess
import threading
from typing import List, Dict

# Configuration for the three Oracle layers
ORACLE_LAYERS = [
    {
        "name": "Primary Consciousness",
        "layer": 1,
        "port": 8765,
        "mode": 1,  # RNG only
        "args": ["--mode", "inference", "--test-rng", "--no-eeg", "--stream-port", "8765", "--stream-id", "oracle_layer_1"]
    },
    {
        "name": "Subconscious",
        "layer": 2, 
        "port": 8766,
        "mode": 2,  # RNG + EEG
        "args": ["--mode", "inference", "--test-rng", "--test-eeg-mode", "stable", "--stream-port", "8766", "--stream-id", "oracle_layer_2"]
    },
    {
        "name": "Universal Consciousness",
        "layer": 3,
        "port": 8767,
        "mode": 2,  # RNG + EEG
        "args": ["--mode", "inference", "--test-rng", "--test-eeg-mode", "stable", "--stream-port", "8767", "--stream-id", "oracle_layer_3"]
    }
]


class InferenceProcess:
    """Manages a single inference process"""
    
    def __init__(self, layer_config: Dict):
        self.config = layer_config
        self.process: subprocess.Popen = None
        self.is_running = False
        
    def start(self):
        """Start the inference process"""
        if self.is_running:
            return
            
        try:
            # Build command
            cmd = [sys.executable, "run.py"] + self.config["args"] + ["--enable-streaming"]
            
            print(f"Starting {self.config['name']} (Layer {self.config['layer']})...")
            print(f"  Command: {' '.join(cmd)}")
            print(f"  Port: {self.config['port']}")
            print(f"  Mode: {self.config['mode']}")
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.is_running = True
            
            # Start output monitoring thread
            self.output_thread = threading.Thread(
                target=self._monitor_output,
                daemon=True
            )
            self.output_thread.start()
            
            print(f"  ✓ Started {self.config['name']} (PID: {self.process.pid})")
            
        except Exception as e:
            print(f"  ✗ Failed to start {self.config['name']}: {e}")
            self.is_running = False
            
    def _monitor_output(self):
        """Monitor process output"""
        if not self.process:
            return
            
        layer_name = self.config['name']
        
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line.strip():
                    print(f"[{layer_name}] {line.strip()}")
                    
        except Exception as e:
            print(f"[{layer_name}] Output monitoring error: {e}")
            
    def stop(self):
        """Stop the inference process"""
        if not self.is_running or not self.process:
            return
            
        try:
            print(f"Stopping {self.config['name']}...")
            
            # Terminate process
            self.process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"  Force killing {self.config['name']}...")
                self.process.kill()
                self.process.wait()
                
            self.is_running = False
            print(f"  ✓ Stopped {self.config['name']}")
            
        except Exception as e:
            print(f"  ✗ Error stopping {self.config['name']}: {e}")
            
    def is_alive(self) -> bool:
        """Check if process is still running"""
        if not self.process:
            return False
        return self.process.poll() is None


class Oracle369MultiLauncher:
    """Manages multiple inference processes for the 369 Oracle"""
    
    def __init__(self):
        self.processes: List[InferenceProcess] = []
        self.is_running = False
        
        # Create processes for each layer
        for layer_config in ORACLE_LAYERS:
            process = InferenceProcess(layer_config)
            self.processes.append(process)
            
    def start_all(self):
        """Start all inference processes"""
        if self.is_running:
            print("Launcher is already running")
            return
            
        print("🔮 Starting 369 Oracle Multi-Inference System")
        print("=" * 60)
        
        # Start each process with delay
        for i, process in enumerate(self.processes):
            process.start()
            
            # Wait between starts to avoid port conflicts
            if i < len(self.processes) - 1:
                time.sleep(2)
                
        self.is_running = True
        
        # Check if all started successfully
        time.sleep(3)
        running_count = sum(1 for p in self.processes if p.is_alive())
        
        print("\n" + "=" * 60)
        print(f"Started {running_count}/{len(self.processes)} inference streams")
        
        if running_count == len(self.processes):
            print("✓ All inference streams are running successfully!")
            print("\nOracle streams available:")
            for process in self.processes:
                config = process.config
                status = "🟢 Running" if process.is_alive() else "🔴 Stopped"
                print(f"  {status} {config['name']}: ws://localhost:{config['port']}")
                
            print("\n🌟 Ready for 369 Oracle consciousness interpretation!")
            print("   Launch the Oracle interface and enable 'Network Mode'")
            
        else:
            print("⚠️  Some inference streams failed to start")
            
    def stop_all(self):
        """Stop all inference processes"""
        if not self.is_running:
            return
            
        print("\n🛑 Stopping all inference streams...")
        
        for process in self.processes:
            process.stop()
            
        self.is_running = False
        print("✓ All inference streams stopped")
        
    def monitor_health(self):
        """Monitor the health of all processes"""
        while self.is_running:
            alive_count = sum(1 for p in self.processes if p.is_alive())
            
            if alive_count < len(self.processes):
                print(f"\n⚠️  Warning: Only {alive_count}/{len(self.processes)} inference streams running")
                
                # Show status of each process
                for process in self.processes:
                    config = process.config
                    status = "Running" if process.is_alive() else "Stopped"
                    print(f"  {config['name']}: {status}")
                    
            time.sleep(10)  # Check every 10 seconds
            
    def get_status(self) -> Dict:
        """Get status of all processes"""
        return {
            "is_running": self.is_running,
            "processes": [
                {
                    "name": p.config["name"],
                    "layer": p.config["layer"],
                    "port": p.config["port"],
                    "is_alive": p.is_alive(),
                    "pid": p.process.pid if p.process else None
                }
                for p in self.processes
            ]
        }


def main():
    """Main launcher function"""
    print("369 Oracle Multi-Inference Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("run.py"):
        print("Error: run.py not found. Please run this script from the consciousness-app directory.")
        sys.exit(1)
        
    # Create launcher
    launcher = Oracle369MultiLauncher()
    
    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nReceived shutdown signal...")
        launcher.stop_all()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start all inference processes
        launcher.start_all()
        
        if launcher.is_running:
            print("\nPress Ctrl+C to stop all inference streams...")
            
            # Start health monitoring in background
            health_thread = threading.Thread(target=launcher.monitor_health, daemon=True)
            health_thread.start()
            
            # Keep running until interrupted
            while launcher.is_running:
                time.sleep(1)
                
                # Check if all processes are still alive
                if not any(p.is_alive() for p in launcher.processes):
                    print("\n💀 All inference processes have died. Exiting...")
                    break
                    
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        
    finally:
        launcher.stop_all()


if __name__ == "__main__":
    main()