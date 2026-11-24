# Bounded attention for long CPU inputs

`serve_bounded.py` launches the original Guard classifier with an additive wrapper
around PyTorch scaled dot-product attention. It uses a serial HTTP server and the
same model loading and command-line options as `serve.py`. Model weights,
float32 precision, tokenizer, complete input, inference heads, and thresholds are
unchanged. Keep the original four-thread CPU launch environment.

For CPU float32 calls with zero dropout and more than 4,096 query positions,
`bounded_sdpa.py` splits the query positions into blocks of 256. Every block sees
the complete original key and value tensors, with the same GQA setting and scale.
For a causal call, a key is allowed exactly when its original position is no
greater than the original query position. The wrapper supplies this explicit
offset mask and disables the kernel's local causal flag. Existing boolean or
additive masks are sliced only along their query dimension; broadcast dimensions
and fully masked rows retain PyTorch's math-kernel semantics. Unsupported mask
shapes are rejected before allocating an attention matrix.

Attention softmax normalizes each query independently over the same full key
sequence, so this operation is mathematically equivalent to unchunked attention.
Each block explicitly uses the math backend. Different matrix shapes or operation
ordering can still change floating-point rounding; bitwise identity is not
claimed, and model-level label/probability parity must be checked separately.
The quadratic score workspace becomes proportional to
`batch * heads * min(chunk_size, queries) * keys`. At eight heads and 8,463 tokens,
one float32 score matrix falls from approximately 2.13 GiB to 66 MiB with a
256-query block. Other model allocations remain, so this is not a fixed cap on
total process memory.

Short calls, other devices or dtypes, and nonzero dropout delegate directly to
the original SDPA function. Set `JEV_SDPA_CHUNK_SIZE` to a positive integer to
change the block size. Set `JEV_SDPA_CHUNK_THRESHOLD=0` to force the bounded path
for short diagnostic samples; its production default is `4096`.

The first recorded service crash was a confirmed system commit exhaustion.
A later recovered long request completed with a large memory peak before a
controlled stop; that observation is not evidence of a second crash. Serial
request handling alone did not prevent the long-input peak. The exact source of
all observed committed memory has not been established.

Serial serving cannot answer concurrent health requests during inference. These
new files do not modify original inference code, historical results, or any live
service. They should be selected explicitly only after parity validation.

The optional notebook adapter is `code/notebook_workflow_bounded.py`. Importing
it does not run or activate a service. Its `run_all` function delegates the six
experiments to the original workflow, sharing the serial adapter's process-wide
single-workflow lock. Automatic startup explicitly uses chunk size 256 and
threshold 4096, overriding diagnostic environment settings. A healthy external
service is reused only when its reported runtime matches these production
settings, CPU float32, four threads, and full-input preservation. A diagnostic
service with threshold zero must therefore be restarted with production settings
before this adapter can reuse it.

Successful health and prediction responses include a `guard_runtime` object with
settings and source hashes. Authorization, routes, errors and prediction values
are handled by the original server. Each new workflow session receives an
exclusively created `guard_runtime.json`, recording the actual observed health
response and package source hashes. The adapter never replaces an existing
provenance file and never stops an external service it did not start.

`code/test_bounded_workflow.py` checks these behaviors with the original HTTP
handler and mocked classifier, process and job operations. It does not load model
weights, access an API, or start or stop a real service.
