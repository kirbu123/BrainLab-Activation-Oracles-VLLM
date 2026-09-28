# Activation injection

How this repo copies residual activations from a **target VLM forward** and writes them into an **oracle** forward. Code: `nl_probes/utils/dataset_utils.py` (prefix + `?` slots), `nl_probes/utils/steering_hooks.py` (write), `nl_probes/utils/activation_utils.py` (read).

The oracle never sees the target image. The only link is a tensor \(h^{\mathrm{src}}\) added at reserved token positions.

---

## 1. Oracle prompt prefix

`SPECIAL_TOKEN = " ?"` (space + question mark). `tokenizer.encode(" ?", add_special_tokens=False)` must be **one** id.

`get_introspection_prefix(source_layers, K)` builds:

```
Layer: ℓ₀
Layer: ℓ₁
…
 ? ? … ? \n
```

- One `Layer: ℓ\n` line per **source** layer (the VLM layer the vector came from).
- Then **K** copies of `" ?"`, then space and newline.
- \(K\) = number of copied source positions (`num_positions`).

`create_training_datapoint` concatenates this prefix with `oracle_question`, wraps it as a user turn, applies the chat template, then appends the GT assistant string. CE labels are \(-100\) on the prompt; loss is only on the answer.

`find_pattern_in_tokens` walks `input_ids` and records the first \(K\) occurrences of the `?` token id. They must be consecutive. The two tokens after the last `?` must decode to a string containing `\n`. Those indices are `TrainingDataPoint.positions` — the injection slots.

Default dest layers: `hook_onto_layers = [1, …, N]` (`injection_dest_layers`, flag `--num-injection-layers`). Dest layer index is the **oracle decoder** layer, not the source percent.

---

## 2. Reading activations (target copy)

If `steering_vectors` is `None`, `materialize_missing_steering_vectors` runs the **same** HF model with **`disable_adapter()`** (base VLM, no oracle LoRA) on `context_input_ids` / VLM `target_messages` + image.

At each source layer \(\ell \in \{25\%,50\%,75\%\}\) of the decoder residual:

\[
h^{\mathrm{src}}_{k}
= \mathrm{resid}^{(\ell)}[\texttt{context\_positions}[k]]
\in \mathbb{R}^{d}
\quad k=0,\ldots,K-1
\]

Stored as `[K, d]`. For \(N>1\) dests, `dest_steering_vectors[i]` is the vector from source layer `source_layers[i]`, still length \(K\).

Optional DeepStack: encoder visual features aligned onto `?` (or visual tokens) when `--deepstack-injection` is on. That is a second write group, not the `?` prefix itself.

---

## 3. Writing into the oracle (the actual injection)

`oracle_steering_hooks` registers a **forward hook** on decoder layer(s) \(1..N\). During the oracle forward, after that layer’s residual is computed, `_apply_steering_writes` runs **only if sequence length \(L>1\)** (prompt pass, not the single-token decode step).

Let \(p_k\) be the \(k\)-th `?` index in the **oracle** sequence. Let \(h_{p_k}\) be the residual at that slot **before** the write. Let \(v_k = h^{\mathrm{src}}_k\).

Implementation (`steering_hooks.py`):

\[
\hat{v}_k = \frac{v_k}{\lVert v_k \rVert_2}
\qquad
\Delta_k = c \cdot \lVert h_{p_k} \rVert_2 \cdot \hat{v}_k
\qquad
h'_{p_k} = h_{p_k} + \mathrm{stopgrad}(\Delta_k)
\]

- \(c\) is `steering_coefficient` (default \(1.0\) per dest; `--optimize-steering-coefs` can search a list).
- Vectors are L2-normalized on the last dim (`F.normalize`).
- Scale matches the **current** residual norm at the `?` (so magnitude follows the oracle stream, direction follows the target).
- `detach_write=True` for decoder writes: \(\Delta\) does not backprop into \(v\) or into \(h\) through the scale; LoRA still trains on tokens after the slots.

The hook docstring says “replace”; the code is **addition**, not overwrite:

```python
resid_BLD[b, pos_t, :] = resid_BLD[b, pos_t, :] + steered_KD.detach()
```

For inj \(N\): dest layer \(i\) gets `dest_steering_vectors[i]` at the **same** `?` positions, coefficient `steering_coefficients[i]`.

If DeepStack and decoder write hit the same module, `get_hf_multi_source_steering_hook` sums two \(\Delta\) computed from a **clone** of the pre-write residual (norms are not taken after the first write).

---

## 4. End-to-end

```
target VLM (adapter off):  [image + target text]
        → resid at context_positions, layer ℓ
        → v_k

oracle (LoRA on):  user = "Layer: ℓ\n ? ? … ? \n" + question
        → locate ? ids → p_k
        → at decoder layer 1..N:  h_{p_k} ← h_{p_k} + c ‖h_{p_k}‖  v̂_k
        → generate / CE on GT
```

The `?` tokens are ordinary embeddings until the hook. Injection is not a special tokenizer token type; it is a post-layer residual add at those indices.
