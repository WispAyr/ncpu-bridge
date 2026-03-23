"""Neural Graphics — framebuffer and drawing primitives through nCPU.

A framebuffer renderer where every pixel calculation is neural:
- Bresenham's line algorithm (neural ADD/SUB/CMP for each step)
- Rectangle fill with neural bounds checking
- Circle drawing (midpoint algorithm, neural arithmetic)
- Text rendering (bitmap font, neural addressing)
- ASCII art output (renders to terminal)

Usage:
    python -m bridge.neural_gfx demo
"""

from __future__ import annotations

import sys
from pathlib import Path

NCPU_PATH = Path("/Users/noc/projects/nCPU")
if str(NCPU_PATH) not in sys.path:
    sys.path.insert(0, str(NCPU_PATH))

from bridge.compute import NCPUBridge

# 3x5 bitmap font for digits and letters
FONT = {
    'N': [0b101, 0b111, 0b111, 0b111, 0b101],
    'C': [0b111, 0b100, 0b100, 0b100, 0b111],
    'P': [0b111, 0b101, 0b111, 0b100, 0b100],
    'U': [0b101, 0b101, 0b101, 0b101, 0b111],
    'O': [0b111, 0b101, 0b101, 0b101, 0b111],
    'K': [0b101, 0b110, 0b100, 0b110, 0b101],
    '!': [0b010, 0b010, 0b010, 0b000, 0b010],
    ' ': [0b000, 0b000, 0b000, 0b000, 0b000],
}


class NeuralFramebuffer:
    """Framebuffer with neural pixel operations."""
    
    def __init__(self, width: int = 60, height: int = 24):
        self.bridge = NCPUBridge()
        self.width = width
        self.height = height
        self.pixels = [[0] * width for _ in range(height)]
        self._ops = 0
    
    def _op(self):
        self._ops += 1
    
    def _set_pixel(self, x: int, y: int, val: int = 1):
        """Set pixel with neural bounds checking."""
        # Neural: 0 <= x < width
        zf_x, sf_x = self.bridge.cmp(x, 0)
        self._op()
        if sf_x and not zf_x:
            return  # x < 0
        
        zf_xw, sf_xw = self.bridge.cmp(x, self.width)
        self._op()
        if not sf_xw and not zf_xw:
            return  # x >= width
        
        # Neural: 0 <= y < height
        zf_y, sf_y = self.bridge.cmp(y, 0)
        self._op()
        if sf_y and not zf_y:
            return
        
        zf_yh, sf_yh = self.bridge.cmp(y, self.height)
        self._op()
        if not sf_yh and not zf_yh:
            return
        
        self.pixels[y][x] = val
    
    def clear(self):
        """Clear framebuffer."""
        self.pixels = [[0] * self.width for _ in range(self.height)]
    
    def line(self, x0: int, y0: int, x1: int, y1: int):
        """Bresenham's line algorithm — every step is neural.
        
        The classic: no floating point, just ADD/SUB/CMP.
        Perfect for neural ALU.
        """
        dx = self.bridge.sub(x1, x0)
        self._op()
        dy = self.bridge.sub(y1, y0)
        self._op()
        
        # Absolute values
        abs_dx = dx if dx >= 0 else self.bridge.sub(0, dx)
        abs_dy = dy if dy >= 0 else self.bridge.sub(0, dy)
        
        sx = 1 if dx >= 0 else -1
        sy = 1 if dy >= 0 else -1
        
        # Neural comparison: is this more horizontal or vertical?
        zf, sf = self.bridge.cmp(abs_dx, abs_dy)
        self._op()
        
        if zf or not sf:  # abs_dx >= abs_dy (more horizontal)
            err = self.bridge.div(abs_dx, 2)
            self._op()
            x, y = x0, y0
            
            for _ in range(abs_dx + 1):
                self._set_pixel(x, y)
                err = self.bridge.sub(err, abs_dy)
                self._op()
                
                # Neural: err < 0?
                zf_e, sf_e = self.bridge.cmp(err, 0)
                self._op()
                if sf_e:  # err < 0
                    y = self.bridge.add(y, sy)
                    self._op()
                    err = self.bridge.add(err, abs_dx)
                    self._op()
                
                x = self.bridge.add(x, sx)
                self._op()
        else:  # More vertical
            err = self.bridge.div(abs_dy, 2)
            self._op()
            x, y = x0, y0
            
            for _ in range(abs_dy + 1):
                self._set_pixel(x, y)
                err = self.bridge.sub(err, abs_dx)
                self._op()
                
                zf_e, sf_e = self.bridge.cmp(err, 0)
                self._op()
                if sf_e:
                    x = self.bridge.add(x, sx)
                    self._op()
                    err = self.bridge.add(err, abs_dy)
                    self._op()
                
                y = self.bridge.add(y, sy)
                self._op()
    
    def rect(self, x: int, y: int, w: int, h: int, fill: bool = False):
        """Draw rectangle with neural bounds."""
        x2 = self.bridge.add(x, self.bridge.sub(w, 1))
        y2 = self.bridge.add(y, self.bridge.sub(h, 1))
        
        if fill:
            for row in range(y, y + h):
                for col in range(x, x + w):
                    self._set_pixel(col, row)
        else:
            self.line(x, y, x2, y)      # top
            self.line(x, y2, x2, y2)    # bottom
            self.line(x, y, x, y2)      # left
            self.line(x2, y, x2, y2)    # right
    
    def circle(self, cx: int, cy: int, r: int):
        """Midpoint circle algorithm — neural arithmetic.
        
        Uses only ADD/SUB/CMP, no multiplication needed.
        """
        x = r
        y = 0
        # Decision parameter: d = 1 - r
        d = self.bridge.sub(1, r)
        self._op()
        
        while True:
            # Neural: x >= y?
            zf, sf = self.bridge.cmp(x, y)
            self._op()
            if sf and not zf:
                break
            
            # Draw 8 octants
            for px, py in [
                (cx + x, cy + y), (cx - x, cy + y),
                (cx + x, cy - y), (cx - x, cy - y),
                (cx + y, cy + x), (cx - y, cy + x),
                (cx + y, cy - x), (cx - y, cy - x),
            ]:
                self._set_pixel(px, py)
            
            y = self.bridge.add(y, 1)
            self._op()
            
            # Neural: d > 0?
            zf_d, sf_d = self.bridge.cmp(d, 0)
            self._op()
            
            if not sf_d and not zf_d:  # d > 0
                x = self.bridge.sub(x, 1)
                self._op()
                # d += 2*(y-x) + 1
                diff = self.bridge.sub(y, x)
                self._op()
                d = self.bridge.add(d, self.bridge.add(self.bridge.mul(2, diff), 1))
                self._op()
            else:
                # d += 2*y + 1
                d = self.bridge.add(d, self.bridge.add(self.bridge.mul(2, y), 1))
                self._op()
    
    def text(self, x: int, y: int, string: str):
        """Render text using bitmap font — neural addressing."""
        cursor_x = x
        for ch in string.upper():
            glyph = FONT.get(ch)
            if not glyph:
                cursor_x = self.bridge.add(cursor_x, 4)
                self._op()
                continue
            
            for row_idx, row_bits in enumerate(glyph):
                for bit in range(3):
                    # Neural AND to check if bit is set
                    mask = self.bridge.shl(1, 2 - bit)
                    self._op()
                    is_set = self.bridge.bitwise_and(row_bits, mask)
                    self._op()
                    
                    zf, _ = self.bridge.cmp(is_set, 0)
                    self._op()
                    
                    if not zf:
                        px = self.bridge.add(cursor_x, bit)
                        self._op()
                        py = self.bridge.add(y, row_idx)
                        self._op()
                        self._set_pixel(px, py)
            
            cursor_x = self.bridge.add(cursor_x, 4)
            self._op()
    
    def render(self) -> str:
        """Render framebuffer to ASCII art."""
        chars = {0: ' ', 1: '█', 2: '▓', 3: '░'}
        lines = []
        for row in self.pixels:
            line = ''.join(chars.get(p, '█') for p in row)
            lines.append(line)
        return '\n'.join(lines)
    
    def render_compact(self) -> str:
        """Render using half-block characters (2 rows per line)."""
        lines = []
        for y in range(0, self.height, 2):
            line = []
            for x in range(self.width):
                top = self.pixels[y][x] if y < self.height else 0
                bot = self.pixels[y + 1][x] if y + 1 < self.height else 0
                
                if top and bot:
                    line.append('█')
                elif top:
                    line.append('▀')
                elif bot:
                    line.append('▄')
                else:
                    line.append(' ')
            lines.append(''.join(line))
        return '\n'.join(lines)


# ── CLI ──

def demo():
    print("Neural Graphics Engine")
    print("=" * 62)
    print("Every pixel calculation → Bresenham/midpoint via neural ALU\n")
    
    fb = NeuralFramebuffer(60, 20)
    
    # Draw border
    fb.rect(0, 0, 60, 20)
    
    # Draw text
    fb.text(3, 2, "NCPU OK!")
    
    # Draw lines (X pattern)
    fb.line(35, 3, 55, 17)
    fb.line(55, 3, 35, 17)
    
    # Draw circle
    fb.circle(20, 13, 5)
    
    print(fb.render())
    print()
    print(f"Neural ops: {fb._ops}")
    print(f"Canvas: {fb.width}×{fb.height} = {fb.width * fb.height} pixels")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    else:
        print("Usage: python -m bridge.neural_gfx [demo]")


if __name__ == "__main__":
    main()
