import time

class RNGSource:
    def __init__(self, device_path: str):
        self.device_path = device_path
        # open USB, etc.

    def read_chunk(self, ms_window: int = 50):
        """
        Read raw bytes from TrueRNG for ms_window milliseconds.
        Return normalized float in [0,1].
        """
        # pseudo-code
        start = time.time()
        raw_bytes = []
        while (time.time() - start) * 1000 < ms_window:
            # read from device
            raw_bytes.append(self._read_byte_from_device())
        if not raw_bytes:
            return 0.5
        avg_val = sum(raw_bytes) / (len(raw_bytes) * 255.0)
        return avg_val

    def triple_sample(self):
        """Return (rA, rB, rC) = 3 consecutive windows for triad spread."""
        return (
            self.read_chunk(50),
            self.read_chunk(50),
            self.read_chunk(50),
        )

    def _read_byte_from_device(self):
        # TODO hardware read
        return 127
