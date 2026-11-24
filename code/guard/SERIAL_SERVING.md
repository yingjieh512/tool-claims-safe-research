# Serial Guard startup adapter

`serve_serial.py` uses the original inference implementation with Python's
single-threaded HTTP server. It keeps one persistent request thread and preserves
the model, checkpoint, tokenizer, input, CPU float32 arithmetic, and four-thread
runtime configuration. Serial serving alone does not solve the observed
long-sequence memory exhaustion.

A new report notebook can import `run_all` from `notebook_workflow_serial`.
That additive adapter delegates all six experiments to the original workflow,
temporarily replaces only its startup and owned-process cleanup hooks, and
restores both hooks on every exit. Run only one workflow per process, including
direct calls to the original workflow, because the hooks are process-wide.
On Windows cleanup targets only the workflow-owned PID and its child process
tree. An already-running external Guard service is reused and left running.

The serial service cannot answer a health request while a prediction is running.
A health-check timeout during inference therefore does not establish a crash.
The original code and historical results are preserved. This adapter changes
neither sampling, retry policy, thresholds, scoring, nor result locations.
