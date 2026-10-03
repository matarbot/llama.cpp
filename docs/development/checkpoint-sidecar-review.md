# Self-review: persisting slot context checkpoints to a sidecar

Self-critical review of the changes in this branch, from the perspective of a
maintainer evaluating the diff. Filed here so that the design contract, safety
analysis, and test map are part of the change itself rather than scattered
through discussion.

Refs: #25913 (the bug this fixes), #16382 (checkpoint requirement extended to
hybrid/recurrent), #16391 (in-memory prompt cache, which persists checkpoints
correctly - the disk path never caught up).

## What the change does

Context checkpoints are the only way the server can roll back non-rewindable
memory (SWA windows, recurrent/hybrid state) to a prefix reuse point. Slot save
serializes tokens + KV cells only, and restore calls `prompt.clear()` - so the
checkpoint ledger never survives the round-trip and every request after a restore
re-processes its whole prefix (silently, at default verbosity).

This branch persists the ledger to a sidecar file (`<save-file>.ckpt`) written
after a successful slot save, and rebuilds the ledger from it at restore. Any
sidecar the reader cannot fully verify - missing, truncated, wrong version, or
saved by a different model topology - is treated as absent: the ledger stays
empty and the slot degrades to full re-processing, byte-identical to the
pre-change behavior.

## Sidecar format (v1)

All integers native-endian, matching `llama_state_seq_save_file`'s existing
assumption (no endian marker anywhere in the main state format; the version
field is the hook if heterogeneous restore ever matters).

```
header:  magic(4) as "LCKP", version(u32) = 1, n_checkpoints(u32), desc_tgt_len(u32), desc_dft_len(u32)
         desc_tgt bytes + desc_dft bytes
record:  n_tokens(i64) id_task(i32) pos_min(i32) pos_max(i32)
         data_tgt_size(u64) + bytes
         data_dft_size(u64) + bytes
         data_spec_size(u64) + bytes
```

Design points, each answering a risk a reviewer should probe:

- **Topology echo.** The header carries `llama_model_desc()` of the target
  context and, when present, the draft context - the topology that produced the
  opaque `data_tgt`/`data_dft` blobs. Load compares both against the live
  contexts and rejects any mismatch before touching KV, so restoring a sidecar
  written by a differently-configured server (different `--model-draft`,
  changed recurrent topology) degrades instead of surfacing deep inside
  `state_seq_set_data`.
- **Atomic publish.** The payload is serialized in memory, written to
  `.ckpt.tmp`, then `rename()`d into place. A save can therefore never leave a
  fresh KV file paired with a half-written or stale ledger. A failed sidecar
  write warns and does not fail the save: the KV save itself succeeded, and
  the degraded-but-consistent state (no checkpoints) is the documented
  fallback.
- **Clamped parsing.** `n_checkpoints` and every blob size are checked against
  the remaining file bytes before any allocation, and the reader requires the
  byte count to land exactly on EOF. A corrupted header cannot drive oversized
  allocations or a runaway loop.
- **Empty ledger.** A save with zero checkpoints still emits a valid
  header-only sidecar. Absence and emptiness are both defined as "no
  checkpoints," so the two readers agree by construction.

## Test coverage

Folded into `tools/server/tests/unit/test_slot_save.py` - no new test file -
reusing the tinygemma3 fixture pattern already established by the multimodal
section of that file (same tiny SWA model, no new model sources, no new
fixtures infrastructure beyond one preset function). The autouse tinyllama2
fixture keeps the pure-attention tests untouched; the checkpoint tests take an
explicit `ckpt_server` fixture, exactly like the mmproj tests take
`mmproj_server`.

Covered:

- in-process save -> restore-into-virgin-slot -> reuse (`cache_n > 0`), with
  `cache_ram = 0` so a hit can only come from disk state, and
  `--checkpoint-min-step 128` so a short prompt actually earns checkpoints;
- save -> **server restart** -> restore -> reuse (the flagship scenario,
  in-memory checkpoint state destroyed with the process);
- sidecar format contract: magic, version, `n_ckpt >= 1`, byte-exact parse of
  the whole ledger, multi-checkpoint fidelity (record count preserved,
  `n_tokens` strictly ordered, every `0 < n_tokens <= n_saved`);
- graceful degradation: sidecar truncated mid-header / mid-descriptor /
  mid-record, then deleted - every restore returns 200 with full `n_restored`,
  and the slot stays deterministic (greedy repeat comparison).

Not covered in CI (and why):

- **hybrid/recurrent models** - the bug's flagship class, but every recurrent
  model in the ecosystem is multi-GB; CI-pinned tinygemma3 is SWA. The guard
  and the sidecar path are identical for both classes (the same
  `common_prompt_checkpoint` records flow through the same serialize/restore
  code); the CI pin constrains model class, not mechanism. Verified
  out-of-band on Qwen3.8-27B (hybrid): restore-then-reuse `prompt_n = 17` /
  0.32 s vs cold 22k tokens / 63 s.
- **draft-context bytes (`data_dft`, `data_spec`)** - the descriptor echo
  covers the draft topology at load; the blob round-trip itself has no
  CI-side coverage because no tiny SWA target has a compatible tiny draft in
  the suite (and `test_slot_save*` had no draft coverage before this change
  either). Verified out-of-band on the same hybrid stack running an MTP draft
  across a backend restart.

## Style notes

- Helpers are file-scope `static` with `server_` prefixes, matching the
  existing pattern in `server-context.cpp` (`server_output_limits`,
  `server_accept_replay`).
- `SRV_WRN`/`SLT_INF`/`SLT_DBG` usage matches the file's conventions; the
  no-sidecar degrade logs at DBG because it is the expected state for legacy
  save files, not an anomaly.
- The sidecar path helper returns `filepath + ".ckpt"` - no glob or pattern
  matching; the filename is server-derived (`fs_validate`-checked upstream of
  the save handler).
- The `utils.py` knob (`checkpoint_min_step`) follows the precedent of
  `cache_ram` immediately above it: optional flag passthrough for a server arg
  the server already supports.

## Commit story

1. `server: add failing tests for slot save/restore checkpoint reuse on SWA models`
   - red tests pinning the contract (they fail on vanilla upstream, which is
   the proof the bug is real and the fix is needed).
2. `server: persist slot context checkpoints to a sidecar file at save` - the
   write side: format v1 with topology echo, atomic publish.
3. `server: rebuild slot checkpoint ledger from sidecar at restore` - the read
   side: validation, clamps, degrade; turns the tests green.
4. `docs: self-review of the checkpoint sidecar change` - this document.
