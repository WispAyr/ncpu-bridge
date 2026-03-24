#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Deploy nCPU Bridge RPC to Bravo (M4 Mac Mini)
# Run from PU2 when Bravo is online
set -e

BRAVO="skynet-bravo@192.168.195.42"
SSH="sshpass -p '1234' ssh -o PubkeyAuthentication=no -o StrictHostKeyChecking=no"
RSYNC="sshpass -p '1234' rsync -avz --exclude __pycache__ --exclude .git --exclude node_modules -e 'ssh -o PubkeyAuthentication=no -o StrictHostKeyChecking=no'"

echo "=== Syncing ncpu-bridge ==="
eval $RSYNC ${SCRIPT_DIR}/../ $BRAVO:~/ncpu-bridge/

echo "=== Syncing nCPU ==="
eval $RSYNC ${NCPU_PATH:-$SCRIPT_DIR/../../nCPU}/ $BRAVO:~/nCPU/

echo "=== Installing deps on Bravo ==="
eval $SSH $BRAVO "pip3 install fastapi uvicorn --break-system-packages 2>/dev/null || pip3 install fastapi uvicorn"

echo "=== Starting server via PM2 ==="
eval $SSH $BRAVO "cd ~/ncpu-bridge/server && pm2 delete ncpu-bridge 2>/dev/null; pm2 start ecosystem.config.cjs && pm2 save"

echo "=== Waiting for startup ==="
sleep 3

echo "=== Testing ==="
curl -s http://192.168.195.42:3950/health | python3 -m json.tool

echo "=== Running benchmark from PU2 ==="
cd ${SCRIPT_DIR}/../server
python3 benchmark.py --url http://192.168.195.42:3950 --n 1000

echo "=== Done ==="
