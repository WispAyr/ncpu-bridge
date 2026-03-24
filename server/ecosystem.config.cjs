module.exports = {
  apps: [{
    name: "ncpu-bridge",
    script: "python3",
    args: "-m uvicorn server:app --host 0.0.0.0 --port 3950",
    cwd: __dirname,
    interpreter: "none",
    env: {
      NCPU_PATH: process.env.HOME + "/nCPU",
      BRIDGE_PATH: process.env.HOME + "/ncpu-bridge",
    },
    max_restarts: 10,
    restart_delay: 2000,
  }],
};
