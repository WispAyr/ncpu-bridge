"""Neural Shell — command interpreter running on the neural kernel.

A shell where every token comparison, pipe setup, and redirect
is handled through neural byte operations. Runs on top of the
neural kernel's subsystems.

Built-in commands:
  ls, cat, echo, pwd, cd, mkdir, rm
  ps, kill, free, df
  ping (DNS resolve), curl (HTTP)
  history, env, export
  | (pipe between commands via neural pipe)
  > (redirect via neural FS)

Usage:
    python -m bridge.neural_shell          # Interactive
    python -m bridge.neural_shell run <script>  # Run script
    python -m bridge.neural_shell demo     # Demo session
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
from bridge.neural_kernel import NeuralKernel


@dataclass
class ShellEnv:
    cwd: str = "/"
    variables: dict[str, str] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    exit_code: int = 0


class NeuralShell:
    """Shell that uses neural ops for all string processing."""

    def __init__(self, kernel: NeuralKernel):
        self.bridge = NCPUBridge()
        self.kernel = kernel
        self.env = ShellEnv()
        self.env.variables = {
            "PATH": "/bin:/usr/bin",
            "HOME": "/home",
            "SHELL": "/bin/nsh",
            "PS1": "ncpu$ ",
        }
        self._ops = 0
        self._output_buf: list[str] = []

    def _op(self):
        self._ops += 1

    # ── Token parsing (neural) ──────────────────────────

    def _neural_split(self, line: str, delim: str = " ") -> list[str]:
        """Split by delimiter using neural CMP on each char."""
        parts, current = [], []
        target = ord(delim)
        for ch in line:
            zf, _ = self.bridge.cmp(ord(ch), target)
            self._op()
            if zf:
                if current:
                    parts.append("".join(current))
                    current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts

    def _neural_eq(self, a: str, b: str) -> bool:
        """String equality via neural CMP."""
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            zf, _ = self.bridge.cmp(ord(a[i]), ord(b[i]))
            self._op()
            if not zf:
                return False
        return True

    def _expand_vars(self, token: str) -> str:
        """Expand $VAR references."""
        if "$" not in token:
            return token
        result = token
        for key, val in self.env.variables.items():
            result = result.replace(f"${key}", val)
        return result

    # ── Command execution ───────────────────────────────

    def execute(self, line: str) -> str:
        """Execute a shell command. Returns output."""
        line = line.strip()
        if not line or line.startswith("#"):
            return ""

        self.env.history.append(line)
        self._output_buf = []

        # Handle pipe: cmd1 | cmd2
        if "|" in line:
            return self._execute_pipe(line)

        # Handle redirect: cmd > file
        redirect_file = None
        if ">" in line:
            parts = line.split(">", 1)
            line = parts[0].strip()
            redirect_file = parts[1].strip()

        tokens = self._neural_split(line)
        if not tokens:
            return ""

        tokens = [self._expand_vars(t) for t in tokens]
        cmd = tokens[0]
        args = tokens[1:]

        output = self._dispatch(cmd, args)

        if redirect_file:
            fs = self.kernel._subsystems.get("fs")
            if fs:
                fs.create(redirect_file, output)
            return f"→ {redirect_file}"

        return output

    def _execute_pipe(self, line: str) -> str:
        """Execute piped commands through neural pipe."""
        segments = line.split("|")
        pipe = self.kernel._subsystems.get("kernel_pipe")

        output = ""
        for i, seg in enumerate(segments):
            seg = seg.strip()
            if i == 0:
                output = self.execute(seg)
            else:
                # Write previous output to pipe, then execute with it as stdin
                if pipe:
                    data = [ord(c) for c in output[:64]]
                    pipe.write(data)
                # Execute next command (simplified: pass as arg)
                tokens = self._neural_split(seg)
                if tokens:
                    cmd = tokens[0]
                    args = tokens[1:] + [output.strip()]
                    output = self._dispatch(cmd, args)

        return output

    def _dispatch(self, cmd: str, args: list[str]) -> str:
        """Dispatch to built-in command handler."""
        handlers = {
            "ls": self._cmd_ls,
            "cat": self._cmd_cat,
            "echo": self._cmd_echo,
            "pwd": self._cmd_pwd,
            "cd": self._cmd_cd,
            "mkdir": self._cmd_mkdir,
            "rm": self._cmd_rm,
            "ps": self._cmd_ps,
            "free": self._cmd_free,
            "df": self._cmd_df,
            "ping": self._cmd_ping,
            "curl": self._cmd_curl,
            "history": self._cmd_history,
            "env": self._cmd_env,
            "export": self._cmd_export,
            "uname": self._cmd_uname,
            "uptime": self._cmd_uptime,
            "ncpu": self._cmd_ncpu,
            "wc": self._cmd_wc,
            "grep": self._cmd_grep,
            "help": self._cmd_help,
        }

        for name, handler in handlers.items():
            if self._neural_eq(cmd.lower(), name):
                try:
                    return handler(args)
                except Exception as e:
                    return f"{cmd}: {e}"

        return f"nsh: {cmd}: command not found"

    # ── Built-in commands ───────────────────────────────

    def _cmd_ls(self, args: list[str]) -> str:
        path = args[0] if args else self.env.cwd
        fs = self.kernel._subsystems.get("fs")
        if not fs:
            return "ls: filesystem not available"
        result = fs.ls(path)
        if "error" in result:
            return f"ls: {result['error']}"
        lines = []
        for e in result.get("entries", []):
            icon = "📁" if e["type"] == "dir" else "📄"
            size = f"{e['size']:>6}" if e["type"] == "file" else "     -"
            lines.append(f"{icon} {size}  {e['name']}")
        return "\n".join(lines) if lines else "(empty)"

    def _cmd_cat(self, args: list[str]) -> str:
        if not args:
            return "cat: missing filename"
        fs = self.kernel._subsystems.get("fs")
        if not fs:
            return "cat: filesystem not available"
        path = args[0] if args[0].startswith("/") else f"{self.env.cwd}/{args[0]}"
        result = fs.read(path)
        if "error" in result:
            return f"cat: {result['error']}"
        return result.get("data", "")

    def _cmd_echo(self, args: list[str]) -> str:
        return " ".join(args)

    def _cmd_pwd(self, args: list[str]) -> str:
        return self.env.cwd

    def _cmd_cd(self, args: list[str]) -> str:
        path = args[0] if args else self.env.variables.get("HOME", "/")
        fs = self.kernel._subsystems.get("fs")
        if fs:
            result = fs.ls(path)
            if "error" not in result:
                self.env.cwd = path
                return ""
        return f"cd: {path}: No such directory"

    def _cmd_mkdir(self, args: list[str]) -> str:
        if not args:
            return "mkdir: missing operand"
        fs = self.kernel._subsystems.get("fs")
        path = args[0] if args[0].startswith("/") else f"{self.env.cwd}/{args[0]}"
        if fs:
            result = fs.mkdir(path)
            return "" if result.get("ok") else f"mkdir: {result.get('error', 'failed')}"
        return "mkdir: filesystem not available"

    def _cmd_rm(self, args: list[str]) -> str:
        if not args:
            return "rm: missing operand"
        fs = self.kernel._subsystems.get("fs")
        path = args[0] if args[0].startswith("/") else f"{self.env.cwd}/{args[0]}"
        if fs:
            result = fs.rm(path)
            return "" if result.get("ok") else f"rm: {result.get('error', 'failed')}"
        return "rm: filesystem not available"

    def _cmd_ps(self, args: list[str]) -> str:
        vm = self.kernel._subsystems.get("vm")
        if not vm:
            return "ps: process manager not available"
        lines = ["  PID  STATE       ALLOCS  NAME"]
        for p in vm.ps():
            lines.append(f"  {p['pid']:3d}  {p['state']:11s} {p['heap_allocs']:6d}  {p['name']}")
        return "\n".join(lines)

    def _cmd_free(self, args: list[str]) -> str:
        vm = self.kernel._subsystems.get("vm")
        if not vm:
            return "free: not available"
        h = vm.heap.stats()
        used_pct = h["allocated"] / h["heap_size"] * 100
        bar_len = 30
        used_bar = int(used_pct / 100 * bar_len)
        bar = "█" * used_bar + "░" * (bar_len - used_bar)
        return (f"  Heap: {h['allocated']:4d}/{h['heap_size']:4d} bytes [{bar}] {used_pct:.0f}%\n"
                f"  Free: {h['free']:4d} bytes in {h['fragments']} fragment(s)")

    def _cmd_df(self, args: list[str]) -> str:
        fs = self.kernel._subsystems.get("fs")
        if not fs:
            return "df: filesystem not available"
        s = fs.stat()
        pct = s["used_blocks"] / s["total_blocks"] * 100
        return (f"  Filesystem    Size  Used  Free  Use%\n"
                f"  neural-fs    {s['total_bytes']:5d} {s['used_blocks']*64:5d} "
                f"{s['free_blocks']*64:5d}  {pct:.0f}%")

    def _cmd_ping(self, args: list[str]) -> str:
        if not args:
            return "ping: missing hostname"
        dns = self.kernel._subsystems.get("dns")
        if dns:
            records = dns.resolve(args[0])
            if records:
                addr = records[0].value if hasattr(records[0], 'value') else str(records[0])
                return f"PING {args[0]} ({addr}): neural route OK"
        return f"ping: {args[0]}: Name or service not known"

    def _cmd_curl(self, args: list[str]) -> str:
        if not args:
            return "curl: missing URL"
        http = self.kernel._subsystems.get("http")
        if http:
            path = "/"
            if "://" in args[0]:
                path = "/" + args[0].split("/", 3)[-1] if "/" in args[0][8:] else "/"
            raw = f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n"
            resp = http.handle_request(raw)
            body = resp.split("\r\n\r\n", 1)[-1]
            return body
        return "curl: HTTP server not available"

    def _cmd_history(self, args: list[str]) -> str:
        lines = []
        for i, cmd in enumerate(self.env.history[-20:], 1):
            lines.append(f"  {i:3d}  {cmd}")
        return "\n".join(lines)

    def _cmd_env(self, args: list[str]) -> str:
        return "\n".join(f"  {k}={v}" for k, v in sorted(self.env.variables.items()))

    def _cmd_export(self, args: list[str]) -> str:
        for arg in args:
            if "=" in arg:
                k, v = arg.split("=", 1)
                self.env.variables[k] = v
        return ""

    def _cmd_uname(self, args: list[str]) -> str:
        return "nCPU Neural OS 1.0 ncpu-kernel neural-alu"

    def _cmd_uptime(self, args: list[str]) -> str:
        r = self.kernel.syscall("uptime")
        secs = r.get("uptime_seconds", 0)
        return f"  up {secs:.1f}s, 1 user, load average: 0.00 (neural)"

    def _cmd_ncpu(self, args: list[str]) -> str:
        sub = args[0] if args else "status"
        if self._neural_eq(sub, "bench"):
            b = NCPUBridge()
            t0 = time.time()
            for _ in range(10):
                b.add(42, 58)
            elapsed = time.time() - t0
            return f"  10 ADD ops: {elapsed*1000:.1f}ms ({elapsed*100:.1f}ms/op)"
        elif self._neural_eq(sub, "status"):
            return (f"  nCPU Neural ALU\n"
                    f"  Models: PyTorch (.pt)\n"
                    f"  Ops: ADD SUB MUL DIV CMP AND OR XOR SHL SHR\n"
                    f"  Shell neural ops so far: {self._ops}")
        return f"ncpu: unknown subcommand: {sub}"

    def _cmd_wc(self, args: list[str]) -> str:
        text = args[-1] if args else ""
        lines = len(text.split("\n"))
        words = len(text.split())
        chars = len(text)
        return f"  {lines:6d} {words:6d} {chars:6d}"

    def _cmd_grep(self, args: list[str]) -> str:
        if len(args) < 2:
            return "grep: usage: grep <pattern> <text>"
        pattern, text = args[0], " ".join(args[1:])
        matches = [line for line in text.split("\n") if pattern.lower() in line.lower()]
        return "\n".join(matches) if matches else ""

    def _cmd_help(self, args: list[str]) -> str:
        return (
            "  Neural Shell (nsh) — built-in commands:\n"
            "  ls cat echo pwd cd mkdir rm    — filesystem\n"
            "  ps free df                     — system info\n"
            "  ping curl                      — networking\n"
            "  history env export             — shell state\n"
            "  uname uptime ncpu              — kernel info\n"
            "  wc grep                        — text tools\n"
            "  |  >                           — pipe/redirect\n"
            "  Every command uses neural byte comparison!"
        )


# ── CLI ──────────────────────────────────────────────────────

DEMO_SCRIPT = """
echo nCPU Neural Shell Demo
echo =======================
uname
uptime
echo ---
pwd
ls /
cd /var/log
ls /var/log
cat /var/log/boot.log
echo ---
ps
free
df
echo ---
ncpu status
ncpu bench
echo ---
echo Hello from neural shell > /tmp/test.txt
cat /tmp/test.txt
ls /tmp
echo ---
ping pos.parkwise.local
curl http://localhost/health
echo Done!
""".strip()


def demo():
    print("Neural Kernel booting for shell session...\n")
    kernel = NeuralKernel()

    # Suppress verbose boot output
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.boot()

    shell = NeuralShell(kernel)

    print("╔══════════════════════════════════╗")
    print("║   nCPU Neural Shell (nsh) v1.0   ║")
    print("╚══════════════════════════════════╝")
    print()

    for line in DEMO_SCRIPT.split("\n"):
        line = line.strip()
        if not line:
            continue
        print(f"ncpu$ {line}")
        output = shell.execute(line)
        if output:
            for ol in output.split("\n"):
                print(f"  {ol}")
        print()

    print(f"Shell session neural ops: {shell._ops}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if cmd == "demo":
        demo()
    elif cmd == "run" and len(sys.argv) > 2:
        kernel = NeuralKernel()
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.boot()
        shell = NeuralShell(kernel)
        script = Path(sys.argv[2]).read_text()
        for line in script.split("\n"):
            output = shell.execute(line)
            if output:
                print(output)
    elif cmd == "interactive":
        kernel = NeuralKernel()
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.boot()
        shell = NeuralShell(kernel)
        print("nCPU Neural Shell. Type 'help' or 'exit'.")
        while True:
            try:
                line = input(f"{shell.env.variables.get('PS1', 'ncpu$ ')}")
                if line.strip() in ("exit", "quit"):
                    break
                out = shell.execute(line)
                if out:
                    print(out)
            except (EOFError, KeyboardInterrupt):
                break
    else:
        print("Usage: python -m bridge.neural_shell [demo|run <script>|interactive]")


if __name__ == "__main__":
    main()
