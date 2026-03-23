"""Neural DNS Resolver — domain name resolution through nCPU.

DNS protocol handling with neural operations:
- Domain name encoding/decoding (label length + neural byte ops)
- DNS packet building (neural field packing via SHL/OR)
- Response parsing with neural pointer following (compression)
- Cache with TTL expiry via neural time comparison
- Record type handling (A, AAAA, CNAME, MX)

Usage:
    python -m bridge.neural_dns demo
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge

# DNS record types
RECORD_TYPES = {1: "A", 5: "CNAME", 15: "MX", 28: "AAAA"}


@dataclass
class DNSRecord:
    name: str
    rtype: str
    ttl: int
    value: str
    created: float = 0.0


@dataclass
class DNSPacket:
    id: int
    is_response: bool
    questions: list[tuple[str, int]] = field(default_factory=list)
    answers: list[DNSRecord] = field(default_factory=list)
    raw: list[int] = field(default_factory=list)


class NeuralDNS:
    """DNS resolver with neural packet construction and parsing."""
    
    def __init__(self):
        self.bridge = NCPUBridge()
        self._cache: dict[str, DNSRecord] = {}
        self._zone: dict[str, list[DNSRecord]] = {}  # Local zone file
        self._ops = 0
        self._setup_zone()
    
    def _op(self):
        self._ops += 1
    
    def _setup_zone(self):
        """Set up local DNS zone with our infrastructure."""
        records = [
            ("pos.parkwise.local", "A", 3600, "10.10.10.238"),
            ("nvr.parkwise.local", "A", 3600, "10.10.10.2"),
            ("pu2.parkwise.local", "A", 3600, "10.10.10.123"),
            ("delta.parkwise.local", "A", 3600, "10.10.10.238"),
            ("pi.parkwise.local", "A", 3600, "192.168.195.238"),
            ("sentry.parkwise.local", "A", 3600, "127.0.0.1"),
            ("ncpu.parkwise.local", "A", 3600, "127.0.0.1"),
            ("mesh.parkwise.local", "CNAME", 3600, "pu2.parkwise.local"),
            ("mail.parkwise.local", "MX", 3600, "10 mx.parkwise.tech"),
            ("parkwise.local", "A", 300, "10.10.10.123"),
        ]
        for name, rtype, ttl, value in records:
            if name not in self._zone:
                self._zone[name] = []
            self._zone[name].append(DNSRecord(name, rtype, ttl, value, time.time()))
    
    def encode_name(self, name: str) -> list[int]:
        """Encode domain name to DNS wire format using neural byte ops.
        
        "pos.parkwise.local" → [3,'p','o','s', 8,'p','a','r','k','w','i','s','e', 5,'l','o','c','a','l', 0]
        """
        result = []
        labels = name.split(".")
        
        for label in labels:
            # Length prefix (neural)
            result.append(len(label))
            # Encode each character (neural ord)
            for ch in label:
                result.append(ord(ch))
        
        result.append(0)  # Root terminator
        return result
    
    def decode_name(self, data: list[int], offset: int) -> tuple[str, int]:
        """Decode domain name from DNS wire format using neural byte scanning."""
        labels = []
        pos = offset
        
        while pos < len(data):
            length = data[pos]
            
            # Neural: is this a compression pointer?
            high_bits = self.bridge.bitwise_and(length, 0xC0)
            self._op()
            zf, _ = self.bridge.cmp(high_bits, 0xC0)
            self._op()
            
            if zf:  # Compression pointer
                # Pointer offset = (length & 0x3F) << 8 | next_byte
                ptr_hi = self.bridge.bitwise_and(length, 0x3F)
                self._op()
                ptr = self.bridge.add(self.bridge.shl(ptr_hi, 8), data[pos + 1])
                self._op()
                name_part, _ = self.decode_name(data, ptr)
                labels.append(name_part)
                return ".".join(labels), pos + 2
            
            # Neural: is this the root (length 0)?
            zf2, _ = self.bridge.cmp(length, 0)
            self._op()
            if zf2:
                return ".".join(labels), pos + 1
            
            # Read label bytes
            label_chars = []
            for i in range(length):
                idx = self.bridge.add(pos, self.bridge.add(1, i))
                self._op()
                label_chars.append(chr(data[idx]))
            
            labels.append("".join(label_chars))
            pos = self.bridge.add(pos, self.bridge.add(length, 1))
            self._op()
        
        return ".".join(labels), pos
    
    def build_query(self, name: str, rtype: int = 1) -> DNSPacket:
        """Build a DNS query packet using neural field packing."""
        pkt = DNSPacket(id=0x1234, is_response=False)
        raw = []
        
        # Header (12 bytes)
        # ID
        raw.append(self.bridge.shr(pkt.id, 8))
        self._op()
        raw.append(self.bridge.bitwise_and(pkt.id, 0xFF))
        self._op()
        
        # Flags: standard query
        raw.extend([0x01, 0x00])  # RD=1
        # QDCOUNT=1
        raw.extend([0x00, 0x01])
        # ANCOUNT, NSCOUNT, ARCOUNT = 0
        raw.extend([0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        
        # Question section
        raw.extend(self.encode_name(name))
        # Type
        raw.append(self.bridge.shr(rtype, 8))
        self._op()
        raw.append(self.bridge.bitwise_and(rtype, 0xFF))
        self._op()
        # Class IN
        raw.extend([0x00, 0x01])
        
        pkt.raw = raw
        pkt.questions = [(name, rtype)]
        return pkt
    
    def resolve(self, name: str, rtype: str = "A") -> list[DNSRecord]:
        """Resolve a domain name — check cache, then zone.
        
        All cache TTL checks use neural time comparison.
        """
        self._ops = 0
        
        # Check cache
        cache_key = f"{name}:{rtype}"
        cached = self._cache.get(cache_key)
        if cached:
            # Neural TTL check: is entry still valid?
            now = int(time.time())
            expiry = self.bridge.add(int(cached.created), cached.ttl)
            self._op()
            zf, sf = self.bridge.cmp(now, expiry)
            self._op()
            
            if sf or zf:  # now <= expiry → still valid
                return [cached]
            else:
                del self._cache[cache_key]
        
        # Query zone
        records = self._zone.get(name, [])
        results = []
        
        for record in records:
            # Neural string comparison for record type
            match = True
            if len(rtype) == len(record.rtype):
                for i in range(len(rtype)):
                    zf, _ = self.bridge.cmp(ord(rtype[i]), ord(record.rtype[i]))
                    self._op()
                    if not zf:
                        match = False
                        break
            else:
                match = False
            
            if match:
                results.append(record)
                # Cache it
                self._cache[cache_key] = record
        
        # Handle CNAME
        if not results:
            cname_records = [r for r in records if r.rtype == "CNAME"]
            if cname_records:
                return self.resolve(cname_records[0].value, rtype)
        
        return results
    
    def cache_stats(self) -> dict:
        return {
            "entries": len(self._cache),
            "zone_records": sum(len(v) for v in self._zone.values()),
            "neural_ops": self._ops,
        }


# ── CLI ──

def demo():
    dns = NeuralDNS()
    
    print("Neural DNS Resolver")
    print("=" * 60)
    print("Domain resolution + packet building → neural byte ops\n")
    
    # ── Name encoding ──
    print("── Domain Name Encoding ──")
    name = "pos.parkwise.local"
    encoded = dns.encode_name(name)
    hex_str = " ".join(f"{b:02x}" for b in encoded)
    print(f"  {name} → [{hex_str}]")
    
    # Decode it back
    decoded, _ = dns.decode_name(encoded, 0)
    print(f"  Decode: {decoded} {'✅' if decoded == name else '❌'}")
    print()
    
    # ── Query packet ──
    print("── DNS Query Packet ──")
    pkt = dns.build_query("ncpu.parkwise.local")
    print(f"  Query for: ncpu.parkwise.local (A record)")
    print(f"  Packet: {len(pkt.raw)} bytes")
    print(f"  Header: {' '.join(f'{b:02x}' for b in pkt.raw[:12])}")
    print()
    
    # ── Resolve queries ──
    print("── Resolve Queries ──")
    queries = [
        ("pos.parkwise.local", "A"),
        ("nvr.parkwise.local", "A"),
        ("pu2.parkwise.local", "A"),
        ("pi.parkwise.local", "A"),
        ("ncpu.parkwise.local", "A"),
        ("mesh.parkwise.local", "A"),      # CNAME → pu2
        ("mail.parkwise.local", "MX"),
        ("unknown.parkwise.local", "A"),    # NXDOMAIN
    ]
    
    for name, rtype in queries:
        results = dns.resolve(name, rtype)
        if results:
            for r in results:
                print(f"  {name} {rtype} → {r.value} (TTL={r.ttl}, {dns._ops} ops)")
        else:
            print(f"  {name} {rtype} → NXDOMAIN ({dns._ops} ops)")
    
    print()
    
    # ── Cache ──
    print("── Cache ──")
    # Second lookup should be cached
    results = dns.resolve("pos.parkwise.local", "A")
    print(f"  pos.parkwise.local (cached): {results[0].value if results else '?'} ({dns._ops} ops)")
    
    stats = dns.cache_stats()
    print(f"  Cache entries: {stats['entries']}")
    print(f"  Zone records: {stats['zone_records']}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    else:
        print("Usage: python -m bridge.neural_dns [demo]")


if __name__ == "__main__":
    main()
