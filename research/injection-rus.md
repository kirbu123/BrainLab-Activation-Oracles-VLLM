# Activation injection

Как в этом репозитории копируют residual-активации с **форварда целевой VLM** и пишут их в **форвард оракула**. Код: `nl_probes/utils/dataset_utils.py` (префикс и слоты `?`), `nl_probes/utils/steering_hooks.py` (запись), `nl_probes/utils/activation_utils.py` (чтение).

Оракул **не** видит картинку целевой модели. Связь — только тензор \(h^{\mathrm{src}}\), который добавляют в reserved-позиции токенов.

---

## 1. Префикс промпта оракула

`SPECIAL_TOKEN = " ?"` (пробел + вопросительный знак). `tokenizer.encode(" ?", add_special_tokens=False)` обязан дать **один** id.

`get_introspection_prefix(source_layers, K)` собирает строку:

```
Layer: ℓ₀
Layer: ℓ₁
…
 ? ? … ? \n
```

- По одной строке `Layer: ℓ\n` на каждый **source**-слой (слой VLM, откуда взят вектор).
- Затем **K** копий `" ?"`, потом пробел и перевод строки.
- \(K\) = число скопированных source-позиций (`num_positions`).

`create_training_datapoint` склеивает этот префикс с `oracle_question`, оборачивает как user-turn, применяет chat template, затем дописывает GT assistant. CE-метки \(-100\) на промпте; лосс только на ответе.

`find_pattern_in_tokens` идёт по `input_ids` и берёт первые \(K\) вхождений id токена `?`. Они должны идти подряд. Два токена после последнего `?` при декоде обязаны содержать `\n`. Эти индексы — `TrainingDataPoint.positions`, слоты инъекции.

Dest-слои по умолчанию: `hook_onto_layers = [1, …, N]` (`injection_dest_layers`, флаг `--num-injection-layers`). Индекс dest — слой **декодера оракула**, не процент source.

---

## 2. Чтение активаций (копия с target)

Если `steering_vectors is None`, `materialize_missing_steering_vectors` гоняет **ту же** HF-модель с **`disable_adapter()`** (базовая VLM, без LoRA оракула) на `context_input_ids` / `target_messages` VLM + картинка.

На каждом source-слое \(\ell \in \{25\%,50\%,75\%\}\) residual декодера:

\[
h^{\mathrm{src}}_{k}
= \mathrm{resid}^{(\ell)}[\texttt{context\_positions}[k]]
\in \mathbb{R}^{d}
\quad k=0,\ldots,K-1
\]

Хранится как `[K, d]`. При \(N>1\) dest: `dest_steering_vectors[i]` — вектор с source-слоя `source_layers[i]`, длина всё ещё \(K\).

Опционально DeepStack: визуальные фичи энкодера, выровненные на `?` (или visual-токены), если включён `--deepstack-injection`. Это вторая группа записей, не сам префикс `?`.

---

## 3. Запись в оракул (сама инъекция)

`oracle_steering_hooks` вешает **forward hook** на decoder-слои \(1..N\). Во время форварда оракула, после residual этого слоя, `_apply_steering_writes` срабатывает **только если длина последовательности \(L>1\)** (проход по промпту, не шаг decode на один токен).

Пусть \(p_k\) — индекс \(k\)-го `?` в **последовательности оракула**. \(h_{p_k}\) — residual в этом слоте **до** записи. \(v_k = h^{\mathrm{src}}_k\).

Имплементация (`steering_hooks.py`):

\[
\hat{v}_k = \frac{v_k}{\lVert v_k \rVert_2}
\qquad
\Delta_k = c \cdot \lVert h_{p_k} \rVert_2 \cdot \hat{v}_k
\qquad
h'_{p_k} = h_{p_k} + \mathrm{stopgrad}(\Delta_k)
\]

- \(c\) — `steering_coefficient` (по умолчанию \(1.0\) на dest; `--optimize-steering-coefs` может подобрать список).
- Векторы L2-нормируются по последней оси (`F.normalize`).
- Масштаб берётся из **текущей** нормы residual в `?` (амплитуда как у потока оракула, направление как у target).
- `detach_write=True` для записей в декодер: \(\Delta\) не течёт в \(v\) и не течёт в \(h\) через норму; LoRA учится на токенах после слотов.

В докстринге хука написано «replace»; в коде это **сложение**, не перезапись:

```python
resid_BLD[b, pos_t, :] = resid_BLD[b, pos_t, :] + steered_KD.detach()
```

Для inj \(N\): dest-слой \(i\) получает `dest_steering_vectors[i]` в **тех же** позициях `?`, коэффициент `steering_coefficients[i]`.

Если DeepStack и decoder-write попадают в один модуль, `get_hf_multi_source_steering_hook` суммирует два \(\Delta\), посчитанных с **клона** residual до записи (норма второго слагаемого не берётся после первого).

---

## 4. Сквозная схема

```
target VLM (adapter off):  [картинка + target-текст]
        → resid в context_positions, слой ℓ
        → v_k

оракул (LoRA on):  user = "Layer: ℓ\n ? ? … ? \n" + вопрос
        → найти id ? → p_k
        → на decoder-слое 1..N:  h_{p_k} ← h_{p_k} + c ‖h_{p_k}‖  v̂_k
        → generate / CE по GT
```

Токены `?` до хука — обычные эмбеддинги. Инъекция — не отдельный тип токена в tokenizer, а post-layer residual add по этим индексам.
