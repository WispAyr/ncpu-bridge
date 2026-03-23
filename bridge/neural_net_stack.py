"""Neural Networking Stack — packet framing, checksums, and flow control through nCPU.

A simplified TCP-like protocol where every operation is neural:
- Sequence numbers: neural ADD for incrementing
- Checksums: neural XOR/ADD for verification
- Window flow control: neural CMP for capacity checks
- Packet assembly/disassembly: neural SHL/OR for bit packing
- RTT estimation: neural SUB/DIV for smoothed averages

This isn't for actual networking — it's a protocol engine that proves
the neural CPU can handle stateful, multi-step packet processing.

Usage:
    python -m bridge.neural_net_stack demo        # Full protocol demo
    python -m bridge.neural_net_stack checksum <hex>  # Compute packet checksum
    python -m bridge.neural_net_stack throughput  # Simulated throughput test
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge


@dataclass
class Packet:
    seq: int
    ack: int
    flags: int  # SYN=1, ACK=2, FIN=4, RST=8, PSH=16
    window: int
    checksum: int
    payload: list[int]
    timestamp: float = 0.0

    # Flag constants
    SYN = 1
    ACK = 2
    FIN = 4
    RST = 8
    PSH = 16


@dataclass
class ConnectionState:
    local_seq: int = 0
    remote_seq: int = 0
    window_size: int = 8
    max_window: int = 32
    inflight: int = 0
    rtt_smooth: int = 100  # ms, smoothed RTT
    state: str = "CLOSED"  # CLOSED, SYN_SENT, ESTABLISHED, FIN_WAIT, CLOSED
    sent_packets: list[Packet] = field(default_factory=list)
    recv_packets: list[Packet] = field(default_factory=list)
    retransmits: int = 0
    total_sent: int = 0
    total_acked: int = 0


class NeuralNetStack:
    """TCP-like protocol engine running on neural ALU.
    
    Every computation — checksums, sequence arithmetic, window
    management, RTT estimation — goes through trained neural nets.
    """
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    # ── Checksum ────────────────────────────────────────
    
    def compute_checksum(self, data: list[int]) -> int:
        """Fletcher-16 style checksum — simple, fast, neural.
        
        sum1 += byte; sum2 += sum1 (mod 255)
        Only uses neural ADD — no table build needed.
        """
        sum1 = 0
        sum2 = 0
        for byte in data:
            sum1 = self.bridge.add(sum1, byte)
            self._op()
            # Modulo 255 approximation: if sum1 >= 255, subtract 255
            zf, sf = self.bridge.cmp(sum1, 255)
            self._op()
            if not sf:  # sum1 >= 255
                sum1 = self.bridge.sub(sum1, 255)
                self._op()
            sum2 = self.bridge.add(sum2, sum1)
            self._op()
            zf, sf = self.bridge.cmp(sum2, 255)
            self._op()
            if not sf:
                sum2 = self.bridge.sub(sum2, 255)
                self._op()
        
        # Combine: (sum2 << 8) | sum1
        high = self.bridge.shl(sum2, 8)
        self._op()
        return self.bridge.add(high, sum1)
    
    def verify_checksum(self, data: list[int], expected: int) -> bool:
        """Verify checksum using neural CMP."""
        actual = self.compute_checksum(data)
        zf, _ = self.bridge.cmp(actual, expected)
        self._op()
        return zf
    
    # ── Packet Assembly ─────────────────────────────────
    
    def build_packet(self, conn: ConnectionState, flags: int, payload: list[int] = None) -> Packet:
        """Assemble a packet with neural-computed fields."""
        payload = payload or []
        
        # Sequence number increment (neural ADD)
        seq = conn.local_seq
        
        # Build header data for checksum: [seq_hi, seq_lo, ack_hi, ack_lo, flags, window, ...payload]
        seq_hi = self.bridge.shr(seq, 8)
        self._op()
        seq_lo = self.bridge.bitwise_and(seq, 0xFF)
        self._op()
        ack_hi = self.bridge.shr(conn.remote_seq, 8)
        self._op()
        ack_lo = self.bridge.bitwise_and(conn.remote_seq, 0xFF)
        self._op()
        
        header_data = [seq_hi, seq_lo, ack_hi, ack_lo, flags, conn.window_size] + payload
        checksum = self.compute_checksum(header_data)
        
        pkt = Packet(
            seq=seq,
            ack=conn.remote_seq,
            flags=flags,
            window=conn.window_size,
            checksum=checksum,
            payload=payload,
            timestamp=time.time(),
        )
        
        # Advance sequence by payload length (neural ADD)
        if payload:
            conn.local_seq = self.bridge.add(conn.local_seq, len(payload))
            self._op()
        elif flags & Packet.SYN or flags & Packet.FIN:
            conn.local_seq = self.bridge.add(conn.local_seq, 1)
            self._op()
        
        conn.sent_packets.append(pkt)
        conn.total_sent = self.bridge.add(conn.total_sent, 1)
        self._op()
        
        return pkt
    
    def validate_packet(self, pkt: Packet) -> dict:
        """Validate a received packet — all checks neural."""
        # Rebuild header data
        seq_hi = self.bridge.shr(pkt.seq, 8)
        seq_lo = self.bridge.bitwise_and(pkt.seq, 0xFF)
        ack_hi = self.bridge.shr(pkt.ack, 8)
        ack_lo = self.bridge.bitwise_and(pkt.ack, 0xFF)
        header_data = [seq_hi, seq_lo, ack_hi, ack_lo, pkt.flags, pkt.window] + pkt.payload
        
        checksum_ok = self.verify_checksum(header_data, pkt.checksum)
        
        return {
            "checksum_valid": checksum_ok,
            "seq": pkt.seq,
            "ack": pkt.ack,
            "flags": self._decode_flags(pkt.flags),
            "payload_len": len(pkt.payload),
            "neural_verified": True,
        }
    
    def _decode_flags(self, flags: int) -> list[str]:
        """Decode flag bits using neural AND."""
        result = []
        for name, bit in [("SYN", 1), ("ACK", 2), ("FIN", 4), ("RST", 8), ("PSH", 16)]:
            v = self.bridge.bitwise_and(flags, bit)
            self._op()
            if v:
                result.append(name)
        return result
    
    # ── Flow Control ────────────────────────────────────
    
    def update_window(self, conn: ConnectionState, acked: int) -> int:
        """AIMD window adjustment — all neural.
        
        On ACK: window = min(window + 1, max_window)  — additive increase
        On loss: window = max(window / 2, 1)           — multiplicative decrease
        """
        if acked > 0:
            # Additive increase
            new_window = self.bridge.add(conn.window_size, 1)
            self._op()
            # Cap at max
            zf, sf = self.bridge.cmp(new_window, conn.max_window)
            self._op()
            if not sf and not zf:  # new > max
                new_window = conn.max_window
            conn.window_size = new_window
            conn.total_acked = self.bridge.add(conn.total_acked, acked)
            self._op()
        else:
            # Multiplicative decrease (loss)
            new_window = self.bridge.div(conn.window_size, 2)
            self._op()
            # Floor at 1
            zf, sf = self.bridge.cmp(new_window, 1)
            self._op()
            if sf or zf:  # new <= 1
                new_window = 1
            conn.window_size = new_window
            conn.retransmits = self.bridge.add(conn.retransmits, 1)
            self._op()
        
        return conn.window_size
    
    def update_rtt(self, conn: ConnectionState, sample_ms: int) -> int:
        """Smoothed RTT: srtt = (7/8)*srtt + (1/8)*sample — neural math.
        
        Approximated as: srtt = srtt - srtt/8 + sample/8
        """
        # srtt/8
        eighth = self.bridge.div(conn.rtt_smooth, 8)
        self._op()
        # srtt - srtt/8 = 7/8 * srtt
        seven_eighths = self.bridge.sub(conn.rtt_smooth, eighth)
        self._op()
        # sample/8
        sample_eighth = self.bridge.div(sample_ms, 8)
        self._op()
        # new srtt
        conn.rtt_smooth = self.bridge.add(seven_eighths, sample_eighth)
        self._op()
        
        return conn.rtt_smooth
    
    # ── Connection Lifecycle ────────────────────────────
    
    def three_way_handshake(self) -> tuple[ConnectionState, ConnectionState]:
        """Simulate TCP 3-way handshake — all neural."""
        client = ConnectionState(local_seq=100, state="CLOSED")
        server = ConnectionState(local_seq=300, state="CLOSED")
        
        # Step 1: Client → SYN
        syn = self.build_packet(client, Packet.SYN)
        client.state = "SYN_SENT"
        
        # Step 2: Server receives SYN, sends SYN+ACK
        server.remote_seq = self.bridge.add(syn.seq, 1)
        self._op()
        syn_ack = self.build_packet(server, Packet.SYN | Packet.ACK)
        server.state = "SYN_RECEIVED"
        
        # Step 3: Client receives SYN+ACK, sends ACK
        client.remote_seq = self.bridge.add(syn_ack.seq, 1)
        self._op()
        ack = self.build_packet(client, Packet.ACK)
        client.state = "ESTABLISHED"
        server.state = "ESTABLISHED"
        
        return client, server
    
    def send_data(self, conn: ConnectionState, data: list[int]) -> list[Packet]:
        """Send data in window-sized chunks — all neural."""
        packets = []
        offset = 0
        
        while offset < len(data):
            # Neural comparison: can we send? (inflight < window)
            zf, sf = self.bridge.cmp(conn.inflight, conn.window_size)
            self._op()
            
            if not sf and not zf:  # inflight >= window — stall
                break
            
            # Chunk size = min(remaining, window - inflight, max_segment=16)
            remaining = self.bridge.sub(len(data), offset)
            self._op()
            available = self.bridge.sub(conn.window_size, conn.inflight)
            self._op()
            
            chunk_size = min(remaining, available, 16)
            chunk = data[offset:offset + chunk_size]
            
            pkt = self.build_packet(conn, Packet.PSH | Packet.ACK, chunk)
            packets.append(pkt)
            
            conn.inflight = self.bridge.add(conn.inflight, len(chunk))
            self._op()
            offset = self.bridge.add(offset, len(chunk))
            self._op()
        
        return packets


# ── CLI ──────────────────────────────────────────────────────

def demo():
    stack = NeuralNetStack()
    
    print("Neural Networking Stack")
    print("=" * 60)
    print("TCP-like protocol where every op is a neural network\n")
    
    # ── Handshake ──
    print("── 3-Way Handshake ──")
    client, server = stack.three_way_handshake()
    print(f"  Client: seq={client.local_seq}, ack={client.remote_seq}, state={client.state}")
    print(f"  Server: seq={server.local_seq}, ack={server.remote_seq}, state={server.state}")
    print(f"  Packets exchanged: {len(client.sent_packets) + len(server.sent_packets)}")
    print()
    
    # ── Validate packets ──
    print("── Packet Validation ──")
    for i, pkt in enumerate(client.sent_packets):
        v = stack.validate_packet(pkt)
        ck = "✅" if v["checksum_valid"] else "❌"
        print(f"  Pkt {i}: seq={v['seq']} flags={v['flags']} checksum={ck}")
    for i, pkt in enumerate(server.sent_packets):
        v = stack.validate_packet(pkt)
        ck = "✅" if v["checksum_valid"] else "❌"
        print(f"  Pkt S{i}: seq={v['seq']} flags={v['flags']} checksum={ck}")
    print()
    
    # ── Data Transfer ──
    print("── Data Transfer ──")
    # Send "Hi!" as byte values (keep short — each byte is many neural ops)
    message = [72, 105, 33]
    print(f"  Sending: {''.join(chr(b) for b in message)} ({len(message)} bytes)")
    
    packets = stack.send_data(client, message)
    print(f"  Packets sent: {len(packets)}")
    for i, pkt in enumerate(packets):
        v = stack.validate_packet(pkt)
        payload_str = ''.join(chr(b) for b in pkt.payload)
        ck = "✅" if v["checksum_valid"] else "❌"
        print(f"    [{i}] seq={pkt.seq} len={len(pkt.payload)} cksum={ck} \"{payload_str}\"")
    print()
    
    # ── Flow Control ──
    print("── AIMD Flow Control ──")
    print(f"  Initial window: {client.window_size}")
    
    # Simulate ACKs growing the window
    for i in range(5):
        w = stack.update_window(client, acked=1)
        print(f"  ACK #{i+1}: window → {w}")
    
    # Simulate a loss
    w = stack.update_window(client, acked=0)
    print(f"  LOSS:    window → {w} (halved)")
    
    # Recovery
    for i in range(3):
        w = stack.update_window(client, acked=1)
        print(f"  ACK #{i+6}: window → {w}")
    print()
    
    # ── RTT ──
    print("── RTT Estimation ──")
    print(f"  Initial SRTT: {client.rtt_smooth}ms")
    samples = [120, 95, 110, 88, 105, 92, 98, 100]
    for sample in samples:
        rtt = stack.update_rtt(client, sample)
        print(f"  Sample {sample}ms → SRTT = {rtt}ms")
    print()
    
    # ── Stats ──
    print("── Neural Stats ──")
    print(f"  Total neural ops: {stack._ops}")
    print(f"  Client packets sent: {client.total_sent}")
    print(f"  Client packets acked: {client.total_acked}")
    print(f"  Retransmits: {client.retransmits}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    
    if cmd == "demo":
        demo()
    elif cmd == "checksum" and len(sys.argv) > 2:
        stack = NeuralNetStack()
        data = [int(x, 16) for x in sys.argv[2:]]
        ck = stack.compute_checksum(data)
        print(f"Checksum: {ck:04x}")
    elif cmd == "throughput":
        stack = NeuralNetStack()
        client, server = stack.three_way_handshake()
        
        # Send 256 bytes
        data = list(range(256))
        t0 = time.time()
        packets = stack.send_data(client, data)
        elapsed = time.time() - t0
        
        total_bytes = sum(len(p.payload) for p in packets)
        print(f"Sent {total_bytes} bytes in {len(packets)} packets")
        print(f"Time: {elapsed:.3f}s")
        print(f"Throughput: {total_bytes/elapsed:.0f} bytes/sec (neural)")
        print(f"Neural ops: {stack._ops}")
    else:
        print("Usage: python -m bridge.neural_net_stack [demo|checksum <hex bytes>|throughput]")


if __name__ == "__main__":
    main()
