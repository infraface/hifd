# JSON envelope (schema version 1.0)

Every prediction file is a JSON object with this envelope:

```json
{
  "task":           "age",
  "source":         "original",
  "estimator":      "mivolo",
  "schema_version": "1.0",
  "predictions":    { "<sample_id>": { ... }, ... }
}
```

- `task` — one of: `age`, `gender`, `ethnicity`, `macro_exp`, `micro_exp`,
  `landmark`, `gaze`, `rppg`, `face_embedding`, `niqe`.
- `source` — `original` for the un-modified images, `deid` for the
  de-identified outputs.
- `estimator` — a free-form string identifying the model.
- `schema_version` — must be exactly `"1.0"` for this release.
- `predictions` — a dict mapping sample IDs to task-specific payloads (below).

Categorical tasks also require a top-level `class_names: [...]` list whose
order matches the per-sample `probs` arrays. rPPG files may include a top-level
`fps` integer.

## Per-task payloads

### `age`

```json
{"value": 32.5}
```

### `gender` / `ethnicity` / `macro_exp` / `micro_exp`

```json
{"probs": [0.92, 0.08]}
```

`probs` must sum to ~1 (off by < 1e-4 is auto-renormalized with a warning);
length must equal the envelope's `class_names`.

### `landmark`

```json
{
  "points": [[10.0, 20.0], [30.0, 40.0], ...],
  "left_eye_center":  [45.0, 50.0],
  "right_eye_center": [75.0, 50.0]
}
```

Original and de-id must have the same number of points per sample.

### `gaze`

```json
{"direction": [0.0, 0.0, 1.0]}
```

A 3-vector. Normalization is applied internally; magnitude doesn't matter.

### `rppg`

```json
{"bvp": [0.1, 0.2, ...], "hr_bpm": 72.0, "fps": 30}
```

`fps` may also be set at the envelope top-level. Original and de-id must
match `fps`; BVP lengths may differ by up to 1 second.

### `face_embedding`

```json
{"embedding": [0.013, -0.241, ...]}
```

D-dimensional vector. Normalization is applied internally.

### `niqe`

```json
{"niqe": 3.42}
```

NIQE is no-reference, so only the de-id file is read.

## Multiple estimators per task

Tasks may have several estimator JSONs (e.g. `face_embedding` typically has
`arcface_*`, `cosface_*`, `adaface_*`). They are listed in
`estimators.yaml`:

```yaml
tasks:
  face_embedding:
    estimators:
      - name: arcface
        orig_json: face_embedding/arcface_orig.json
        deid_json: face_embedding/arcface_deid.json
      - name: cosface
        orig_json: face_embedding/cosface_orig.json
        deid_json: face_embedding/cosface_deid.json
```

## methods.yaml

```yaml
original:
  base_path: utilface
methods:
  - name: MyMethod
    base_path: methods/MyMethod
```

Paths are relative to `--data-dir`.

## Compatibility

A loader that sees a `schema_version` other than `"1.0"` raises `ValueError`.
Bumping the schema is a major-version event.
