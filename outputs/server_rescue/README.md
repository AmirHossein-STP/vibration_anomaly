# Rescue pull from the second GPU server (2026-09-17)

16 arm A runs (8 time + 8 order) from the clean-room rebuild, pulled while
the box was still alive. **These runs are degenerate** -- see
GPU_SERVER_PLAN.md's "REPRODUCTION FAILED" note. They are kept precisely
because they are the evidence for that failure, and because the first
server's artifacts were lost by NOT doing this.

Reproduce the summary with:

    python scripts/analysis/summarize_arms.py outputs/server_rescue
