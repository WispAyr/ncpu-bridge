"""CLI wrapper for Skynet to call nCPU bridge operations."""

import argparse
import json
import sys
import time


def cmd_calculate(args):
    from bridge.compute import NCPUBridge

    bridge = NCPUBridge()
    result = bridge.calculate(args.expression)
    print(result)


def cmd_verify(args):
    from bridge.compute import NCPUBridge

    bridge = NCPUBridge()
    ok = bridge.verify(args.operation, args.a, args.b, args.expected)
    print(json.dumps({"verified": ok, "operation": args.operation, "a": args.a, "b": args.b, "expected": args.expected}))
    sys.exit(0 if ok else 1)


def cmd_health_check(args):
    from bridge.health import HealthComputer

    hc = HealthComputer()
    result = hc.check_threshold(args.value, args.threshold, args.name)
    print(json.dumps(result))
    sys.exit(0 if not result["exceeded"] else 1)


def cmd_obligation_check(args):
    from bridge.obligations import ObligationChecker

    oc = ObligationChecker()
    now = args.now or int(time.time())
    result = oc.check_interval(args.last_run, now, args.interval)
    print(json.dumps(result))
    sys.exit(0 if not result["overdue"] else 1)


def cmd_run(args):
    from bridge.compute import NCPUBridge

    bridge = NCPUBridge()
    if args.file:
        assembly = open(args.file).read()
    else:
        assembly = args.assembly
    if args.gpu:
        result = bridge.run_program_gpu(assembly)
    else:
        result = bridge.run_program(assembly)
    print(json.dumps(result, default=str))


def cmd_benchmark(args):
    from bridge.compute import NCPUBridge

    bridge = NCPUBridge()
    results = bridge.benchmark(iterations=args.iterations)
    for op, data in results.items():
        print(f"{op:>4s}: neural={data['neural_us']:>8.1f} us  native={data['native_us']:>8.3f} us  ratio={data['ratio']}x")


def main():
    parser = argparse.ArgumentParser(
        prog="ncpu-bridge",
        description="Bridge to nCPU's verified neural ALU",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # calculate
    p = sub.add_parser("calculate", help="Evaluate arithmetic expression")
    p.add_argument("expression", help='Expression like "48 * 365"')
    p.set_defaults(func=cmd_calculate)

    # verify
    p = sub.add_parser("verify", help="Verify a computation result")
    p.add_argument("operation", help="Operation name (add, sub, mul, ...)")
    p.add_argument("a", type=int)
    p.add_argument("b", type=int)
    p.add_argument("expected", type=int)
    p.set_defaults(func=cmd_verify)

    # health-check
    p = sub.add_parser("health-check", help="Check value against threshold")
    p.add_argument("--value", type=int, required=True, help="Current value")
    p.add_argument("--threshold", type=int, required=True, help="Threshold limit")
    p.add_argument("--name", default="check", help="Check name")
    p.set_defaults(func=cmd_health_check)

    # obligation-check
    p = sub.add_parser("obligation-check", help="Check if obligation is overdue")
    p.add_argument("--last-run", type=int, required=True, help="Last run epoch timestamp")
    p.add_argument("--interval", type=int, required=True, help="Required interval in seconds")
    p.add_argument("--now", type=int, default=None, help="Current epoch (default: now)")
    p.set_defaults(func=cmd_obligation_check)

    # run
    p = sub.add_parser("run", help="Run assembly program on neural CPU")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="Path to .asm file")
    g.add_argument("--assembly", help="Inline assembly string")
    p.add_argument("--gpu", action="store_true", help="Use Metal GPU compute")
    p.set_defaults(func=cmd_run)

    # benchmark
    p = sub.add_parser("benchmark", help="Neural vs native benchmark")
    p.add_argument("--iterations", type=int, default=100)
    p.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
