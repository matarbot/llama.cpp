import os
import struct

import pytest

from utils import *

#
# Prompt-cache reuse across /slots save/restore on SWA / hybrid-recurrent models.
#
# ref: https://github.com/ggml-org/llama.cpp/issues/25913
#
# On models where the server keeps context checkpoints (SWA without --swa-full, or
# hybrid/recurrent memory), prefix reuse at task launch is gated on a checkpoint that
# can roll the non-rewindable memory back to the reuse point (server-context.cpp,
# the "forcing full prompt re-processing" guard). The on-disk slot save/restore path
# does not persist checkpoints: save() serializes only tokens + KV cells, and restore()
# calls prompt.clear(), which wipes any checkpoints the slot still had. A restored slot
# therefore reports full success (n_restored == n_tokens) but re-processes the entire
# prefix on the next request (cache_n == 0), on every request, forever.
#
# These tests pin the expected behavior: checkpoint state must survive the disk round-trip.
# They run on tinygemma3, an SWA model already used by the multimodal slot-save tests;
# the text-only path with the default SWA cache is the configuration the guard fires on.
#

# sidecar checkpoint file written next to the slot save file:
#   magic(4) version(4) n_checkpoints(4), then per checkpoint:
#   n_tokens(i64) id_task(i32) pos_min(i32) pos_max(i32)
#   data_tgt_size(u64) data_tgt, data_dft_size(u64) data_dft, data_spec_size(u64) data_spec
CKPT_MAGIC = b"LCKP"
CKPT_VERSION = 1

server = ServerPreset.tinygemma3()

# a text prefix long enough to produce a context checkpoint during processing,
# and a different suffix to send after the restore (the reuse point)
PREFIX = (
    "The city of Amsterdam was built on the river Amstel. Its concentric canals "
    "were dug in the seventeenth century during the Dutch Golden Age, when the city "
    "grew rapidly as a center of trade and science. The canal ring, planned as a "
    "unified whole, remains one of the most remarkable urban projects of early modern "
    "Europe. "
) * 4
SUFFIX = "What is the name of the river Amsterdam was built on?"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0
    server.n_ctx = 4096
    server.n_predict = 8
    # disable the in-RAM prompt cache: prefix reuse can then only come from the
    # disk-restored state, which is what these tests are about
    server.cache_ram = 0
    return server


def _completion(prompt: str, id_slot: int):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": id_slot,
        "cache_prompt": True,
        "seed": 42,
    })
    assert res.status_code == 200
    return res


def _save(id_slot: int, filename: str) -> int:
    res = server.make_request("POST", f"/slots/{id_slot}?action=save", data={
        "filename": filename,
    })
    assert res.status_code == 200
    return res.body["n_saved"]


def _restore(id_slot: int, filename: str) -> int:
    res = server.make_request("POST", f"/slots/{id_slot}?action=restore", data={
        "filename": filename,
    })
    assert res.status_code == 200
    return res.body["n_restored"]


#
# Reuse after restore, same server process. The prefix was processed in slot 1 and
# restored into the virgin slot 0; the next request to slot 0 must reuse the restored
# prefix instead of re-processing it. Fails today: cache_n == 0 (see issue #25913).
#
# --checkpoint-min-step is lowered so the prefix actually earns a checkpoint during
# processing (the default spacing of 8192 tokens would create none for a prompt this
# size, and then even a working persist path would have nothing to restore).
#
def test_slot_save_restore_reuse_swa_in_process():
    global server
    server.checkpoint_min_step = 128
    server.start()

    res = _completion(PREFIX, 1)
    total_n = res.body["timings"]["prompt_n"] + res.body["timings"]["cache_n"]
    assert res.body["timings"]["cache_n"] == 0  # cold: everything processed

    n_saved = _save(1, "swa_inproc.bin")
    assert n_saved > 0

    n_restored = _restore(0, "swa_inproc.bin")
    assert n_restored == n_saved

    # slot 0 has never seen this prefix in RAM: reuse can only come from the restored state
    res = _completion(PREFIX + SUFFIX, 0)
    cache_n = res.body["timings"]["cache_n"]
    assert cache_n > 0, "restored prefix was not reused: checkpoints were lost in the disk round-trip"
    assert res.body["timings"]["prompt_n"] < total_n


#
# The flagship scenario (issue #25913 + the flash-image cold-start use case):
# save, RESTART the server (RAM checkpoints are gone), restore, reuse.
# Fails today: cache_n == 0 after a full re-processing of the whole prefix.
#
def test_slot_save_restore_reuse_swa_across_restart():
    global server
    server.checkpoint_min_step = 128
    server.start()

    res = _completion(PREFIX, 0)
    assert res.body["timings"]["cache_n"] == 0

    n_saved = _save(0, "swa_restart.bin")
    assert n_saved > 0

    # restart: all in-memory checkpoint state dies with the process
    server.stop()
    server.start()

    n_restored = _restore(0, "swa_restart.bin")
    assert n_restored == n_saved

    res = _completion(PREFIX + SUFFIX, 0)
    cache_n = res.body["timings"]["cache_n"]
    assert cache_n > 0, "restored prefix was not reused after restart: checkpoints were never persisted"


#
# Contract test for the fix format: saving a slot that holds checkpoints must write a
# sidecar checkpoint file next to the slot file, with the documented magic, version and
# at least one checkpoint record. Fails today: the save handler writes no sidecar.
#
def test_slot_save_writes_checkpoint_sidecar():
    global server
    server.start()

    _completion(PREFIX, 0)
    n_saved = _save(0, "swa_sidecar.bin")

    path = os.path.join("tmp", "swa_sidecar.bin")
    sidecar = path + ".ckpt"
    assert os.path.exists(sidecar), "save did not produce a checkpoint sidecar"

    with open(sidecar, "rb") as f:
        header = f.read(12)
        magic, version, n_ckpt = header[:4], struct.unpack_from("<I", header, 4)[0], struct.unpack_from("<I", header, 8)[0]
        assert magic == CKPT_MAGIC
        assert version == CKPT_VERSION
        assert n_ckpt >= 1, "sidecar holds no checkpoints"

        # the first checkpoint record: n_tokens(i64) id_task(i32) pos_min(i32) pos_max(i32)
        rec = struct.unpack("<qiII", f.read(struct.calcsize("<qiII")))
        first_n_tokens = rec[0]
    assert 0 < first_n_tokens <= n_saved


#
# Forward-compat / degradation test: restoring a save file WITHOUT a checkpoint sidecar
# (a file written by an older server, or a sidecar that was deleted) must still succeed
# and leave the slot usable. It may degrade to full re-processing, but it must not fail
# or corrupt. Expected to be green before and after the fix - this pins the compatibility
# promise of the sidecar format.
#
def test_slot_restore_without_sidecar_degrades_gracefully():
    global server
    server.start()

    _completion(PREFIX, 0)
    n_saved = _save(0, "swa_nosidecar.bin")

    # emulate a legacy save file: remove the sidecar if the fixed server wrote one
    sidecar = os.path.join("tmp", "swa_nosidecar.bin.ckpt")
    if os.path.exists(sidecar):
        os.remove(sidecar)

    n_restored = _restore(1, "swa_nosidecar.bin")
    assert n_restored == n_saved

    # the slot must remain usable after the degraded restore: the request completes and
    # the slot keeps serving repeat requests deterministically (greedy)
    res = _completion(PREFIX + SUFFIX, 1)
    assert res.status_code == 200
    content = res.body["content"]
    res2 = _completion(PREFIX + SUFFIX, 1)
    assert res2.status_code == 200
    assert res2.body["content"] == content


#
# Mechanism control: prefix reuse on this model is checkpoint-dependent even with NO disk
# round-trip involved. With --ctx-checkpoints 0, ordinary two-turn reuse in the SAME slot
# is forced to full re-processing (same guard, reached by a different route). Green today
# and after the fix (the fix persists checkpoints; with checkpointing disabled there is
# nothing to persist). This isolates the checkpoint ledger as the load-bearing mechanism.
#
def test_swa_reuse_requires_checkpoints_control():
    global server
    server.ctx_checkpoints = 0
    server.start()

    res = _completion(PREFIX, 0)
    assert res.body["timings"]["cache_n"] == 0

    res = _completion(PREFIX + SUFFIX, 0)
    assert res.body["timings"]["cache_n"] == 0, "expected no reuse with checkpoints disabled"
