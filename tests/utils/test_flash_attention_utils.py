# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""What the flash-attention wrapper hands the kernel, on CPU and without the `kernels` package.

A fake hub kernel written in plain torch records its arguments, so the whole chain the wrapper builds
(`attn_implementation` -> hub kernel -> `flash_attn_varlen_func`) runs on CPU, and each case is compared with
per-document eager attention.
"""

import types
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from transformers import modeling_flash_attention_utils as fa_utils
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.integrations.hub_kernels import get_attn_kernel_version
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


FA4 = "kernels-community/flash-attn4"
FA2 = "kernels-community/flash-attn2"


class FakeConfig:
    def __init__(self, implementation: str):
        self._attn_implementation = implementation


class FakeAttention(torch.nn.Module):
    """What the wrapper reads off a module: config, is_causal, layer_idx, and a Linear for the dtype check."""

    def __init__(self, implementation: str):
        super().__init__()
        self.config = FakeConfig(implementation)
        self.is_causal = True
        self.layer_idx = 0
        self.proj = torch.nn.Linear(8, 8, bias=False)


def attend(q, k, v, cu_seqlens, causal, softmax_scale):
    """Per-document causal attention with GQA: `(total, heads, dim) -> (total, heads, dim_v)`, like flash."""
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    pieces = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        out = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1)[None],
            k[start:end].transpose(0, 1)[None],
            v[start:end].transpose(0, 1)[None],
            is_causal=causal,
            scale=softmax_scale,
        )
        pieces.append(out[0].transpose(0, 1))
    return torch.cat(pieces, dim=0)


class FakeKernel:
    """A `flash_attn_varlen_func` with flash-attn4's signature that records the arguments it is called with."""

    def __init__(self, calls: list[dict], requires_int_max_seqlen: bool):
        self.calls = calls
        self.requires_int_max_seqlen = requires_int_max_seqlen

    def flash_attn_varlen_func(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        learnable_sink=None,
        softcap=0.0,
        **kwargs,
    ):
        # flash-attn4 checks the types of the two lengths on the host side, before it compiles the kernel.
        if self.requires_int_max_seqlen:
            assert isinstance(max_seqlen_q, int), f"max_seqlen_q must be an int, got {type(max_seqlen_q)}"
            assert isinstance(max_seqlen_k, int), f"max_seqlen_k must be an int, got {type(max_seqlen_k)}"
        self.calls.append(
            {
                "q": tuple(q.shape),
                "v": tuple(v.shape),
                "cu_seqlens": cu_seqlens_q.tolist(),
                "max_seqlen": (max_seqlen_q, max_seqlen_k),
            }
        )
        return attend(q, k, v, cu_seqlens_q.tolist(), causal, softmax_scale)


@pytest.fixture
def kernel_calls():
    """Serve a fake hub kernel for both flash-attn repos, and restore everything the load registers."""
    calls: list[dict] = []

    def get_kernel(repo_id, revision=None, version=None, **kwargs):
        module = types.ModuleType(repo_id)
        module.flash_attn_varlen_func = FakeKernel(calls, "flash-attn4" in repo_id).flash_attn_varlen_func
        module.flash_attn_func = module.flash_attn_varlen_func
        module.loaded_version = version
        return module

    saved = (
        fa_utils._loaded_implementation,
        fa_utils._flash_fn,
        fa_utils._flash_varlen_fn,
        fa_utils._flash_with_kvcache_fn,
        fa_utils._pad_fn,
        fa_utils._unpad_fn,
        fa_utils._process_flash_kwargs_fn,
        fa_utils._process_with_kvcache_kwargs_fn,
    )
    with (
        # transformers gates the hub path on the installed distribution, which a CPU dev env does not have
        patch("transformers.integrations.hub_kernels.is_kernels_available", return_value=True),
        patch("transformers.integrations.hub_kernels.get_kernel", get_kernel),
    ):
        fa_utils._loaded_implementation = None
        try:
            yield calls
        finally:
            (
                fa_utils._loaded_implementation,
                fa_utils._flash_fn,
                fa_utils._flash_varlen_fn,
                fa_utils._flash_with_kvcache_fn,
                fa_utils._pad_fn,
                fa_utils._unpad_fn,
                fa_utils._process_flash_kwargs_fn,
                fa_utils._process_with_kvcache_kwargs_fn,
            ) = saved
            for implementation in (FA4, FA2):
                if implementation in ALL_ATTENTION_FUNCTIONS._global_mapping:
                    del ALL_ATTENTION_FUNCTIONS._global_mapping[implementation]


def packed_inputs(total=6, heads=4, kv_heads=2, dim=8, v_dim=None, lengths=(2, 4)):
    torch.manual_seed(0)
    q = torch.randn(1, heads, total, dim)
    k = torch.randn(1, kv_heads, total, dim)
    v = torch.randn(1, kv_heads, total, v_dim or dim)
    position_ids = torch.cat([torch.arange(length) for length in lengths])[None]
    return q, k, v, position_ids


def per_document_reference(q, k, v, lengths, scaling=0.5):
    """`(batch, tokens, heads, dim_v)`, the layout the attention interface returns."""
    cu = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    q, k, v = (tensor[0].transpose(0, 1) for tensor in (q, k, v))  # (heads, tokens, dim) -> (tokens, heads, dim)
    return attend(q, k, v, cu, True, scaling)[None]


def test_the_hub_flash_attn4_version_is_a_build_that_imports():
    """v0 calls `cutlass.cute.ThrMma`, which nvidia-cutlass-dsl 4.4.0 removed, so `attn_implementation=
    "kernels-community/flash-attn4"` cannot load with it."""
    assert get_attn_kernel_version(FA4) == 1


def test_packed_sequence_hands_flash_attn4_python_ints(kernel_calls):
    """`prepare_fa_kwargs_from_position_ids` returns 0-dim tensors; flash-attn4's host side needs ints."""
    q, k, v, position_ids = packed_inputs(lengths=(2, 4))
    module = FakeAttention(FA4)

    out, weights = flash_attention_forward(module, q, k, v, None, dropout=0.0, scaling=0.5, position_ids=position_ids)

    assert weights is None
    assert kernel_calls == [{"q": (6, 4, 8), "v": (6, 2, 8), "cu_seqlens": [0, 2, 6], "max_seqlen": (4, 4)}]
    torch.testing.assert_close(out, per_document_reference(q, k, v, (2, 4)))


def test_a_dense_mask_on_a_packed_row_keeps_the_cu_seqlens_path(kernel_calls):
    """A 4-D mask next to packed `position_ids` says what `cu_seqlens` says, and must not be read as a padding mask.
    A 2-D padding mask is still used."""
    q, k, v, position_ids = packed_inputs(lengths=(2, 4))
    dense = torch.zeros(1, 1, 6, 6)  # additive, causal, block-diagonal: what a packed eager forward builds

    out, _ = flash_attention_forward(
        FakeAttention(FA4), q, k, v, dense, dropout=0.0, scaling=0.5, position_ids=position_ids
    )
    assert kernel_calls == [{"q": (6, 4, 8), "v": (6, 2, 8), "cu_seqlens": [0, 2, 6], "max_seqlen": (4, 4)}]
    torch.testing.assert_close(out, per_document_reference(q, k, v, (2, 4)))

    q, k, v, _ = packed_inputs(total=6, lengths=(6,))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    out, _ = flash_attention_forward(FakeAttention(FA4), q, k, v, mask, dropout=0.0, scaling=0.5)
    assert kernel_calls[1]["q"] == (4, 4, 8), "the two padding tokens must be dropped before the kernel"
    assert out.shape == (1, 6, 4, 8), "and padded back afterwards"


def test_mla_keeps_the_value_head_width_on_flash_attn4(kernel_calls):
    """MLA's `(q/k 192, v 128)`: flash-attn4 takes the value at its own width, flash-attn2 needs it padded."""
    q, k, v, position_ids = packed_inputs(dim=192, v_dim=128, lengths=(2, 4))

    out, _ = flash_attention_forward(
        FakeAttention(FA4), q, k, v, None, dropout=0.0, scaling=0.5, position_ids=position_ids
    )
    assert kernel_calls[0]["v"] == (6, 2, 128), "padded to 192, flash-attn4 rejects (192, 192)"
    assert out.shape == (1, 6, 4, 128)
    torch.testing.assert_close(out, per_document_reference(q, k, v, (2, 4)))

    padded, _ = flash_attention_forward(
        FakeAttention(FA2), q, k, v, None, dropout=0.0, scaling=0.5, position_ids=position_ids
    )
    assert kernel_calls[1]["v"] == (6, 2, 192), "flash-attn2 wants the value at the query's width"
    torch.testing.assert_close(padded, out)


def test_flash_attn2_still_gets_the_lengths_as_tensors(kernel_calls):
    """Only flash-attn4 needs ints: converting for everyone would add a device sync per layer to flash-attn2."""
    q, k, v, position_ids = packed_inputs(lengths=(2, 4))

    flash_attention_forward(FakeAttention(FA2), q, k, v, None, dropout=0.0, scaling=0.5, position_ids=position_ids)

    assert isinstance(kernel_calls[0]["max_seqlen"][0], torch.Tensor)
    assert int(kernel_calls[0]["max_seqlen"][0]) == 4
