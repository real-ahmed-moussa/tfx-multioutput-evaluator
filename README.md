# MultiOutputEvaluator (TFX Component)
A lightweight and flexible **TFX** component for evaluating **multi-output** TensorFlow models. It computes **per-output** and **global** metrics and writes a **TFMA-compatible** JSON artifact for downstream analysis and reporting.

![PyPI version](https://img.shields.io/pypi/v/tfx-moe.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Downloads](https://static.pepy.tech/badge/tfx-moe)](https://pepy.tech/project/tfx-moe)

---

## 📦 Installation

Install from PyPI:

```bash
pip install tfx-moe
```

---

## ⚙️ Features

- ✅ Evaluate **multi-output** models (e.g., shape `(batch, num_outputs)`)
- ✅ Computes **per-output** and **global** metrics
- ✅ Currently supports **MSE** and **MAE** (easy to extend)
- ✅ **TFMA-compatible** JSON output (`ModelEvaluation` artifact)
- ✅ Works with **TFX Transform** artifacts (`TransformGraph`)
- ✅ Pluggable **dataset input function** (your data, your logic)

---

## 📚 API Reference

### `MultiOutputEvaluator` Component

```python
MultiOutputEvaluator(
            model,                      # Channel[Model]
            examples,                   # Channel[Examples]
            transform_graph,            # Channel[TransformGraph]
            output_names,               # list[str]: names of model outputs in order
            metrics,                    # list[str]: e.g., ['mse', 'mae']
            input_fn_path,              # str: "pkg.mod:func" or "pkg.mod.func"
            input_fn_kwargs=None,       # dict: extra kwargs for input function
            example_split='test'        # str: which split to evaluate
        )
```
### Parameters
1. **model:** `standard_artifacts.Model` – Trained SavedModel (expects predictions of shape `(batch, num_outputs)`).
2. **examples:** `standard_artifacts.Examples` – TFRecords location for the chosen split.
3. **transform_graph:** `standard_artifacts.TransformGraph` – Loaded via tft.`TFTransformOutput`.
4. **output_names** *(list[str])***:** Logical names per output index; order must correspond to your model’s output dimension.
5. **metrics** *(list[str])***:** Currently `['mse', 'mae']` supported.
6. **input_fn_path** *(str)***:** Dotted path to your dataset builder function.
7. **input_fn_kwargs** *(dict, optional)***:** Keyword args passed to the dataset function.
8. **example_split** *(str, default 'test')***:** Split folder name under `Examples` artifact (e.g., `'Split-test`' if that’s how your pipeline writes splits).

---

## 🚀 Quickstart (TFX Usage)

### [1] Add Component to your Pipeline and Run
```python
from tfx_moe import MultiOutputEvaluator

# InteractiveContext Pipeline Runner
multi_output_evaluator = MultiOutputEvaluator(
    model=trainer.outputs['model'],
    examples=transform.outputs["transformed_examples"],
    transform_graph=transform.outputs['transform_graph'],
    
    output_names=OUTPUT_KEYS,                     # e.g., ["y_loc1", "y_loc2", ...]
    metrics=['mse', 'mae'],

    input_fn_path="my_pkg.mdata:input_fn",        # Your dataset builder function
    example_split='Split-test'                    # Match your artifact split naming
)

# Run (InteractiveContext example)
context.run(multi_output_evaluator)

# Access TFMA-like JSON results
evaluation_uri = multi_output_evaluator.outputs['evaluation'].get()[0].uri
print(evaluation_uri)
```

### [2] Parse Results (per-output + global)
```python
import json
import pandas as pd
import tensorflow as tf

with tf.io.gfile.GFile(evaluation_uri, 'r') as f:
    data = json.load(f)                     # TFMA-style dict

output_names = []
mse_values = []
mae_values = []

for key, value in data['metrics'].items():
    if key.endswith('>>mse'):
        output_name = key.split('>>')[0]         # 'per_output>>{name}>>mse' → '{name}'
        output_names.append(output_name)
        mse_values.append(value)
    elif key.endswith('>>mae'):
        mae_values.append(value)

df = pd.DataFrame({
    'Output Name': output_names,
    'MSE': mse_values,
    'MAE': mae_values
})


last_row_index = df.index[-1]
global_mse = df["MSE"].iloc[last_row_index]
global_mae = df["MAE"].iloc[last_row_index]
df = df.drop(index=[last_row_index])              # keep per-output rows only
```

### [3] Global Metrics
```python
print(f'Global MSE {global_mse}')
print(f'Global MAE {global_mae}')
```

### [4] Aggregate Across Outputs
```python
global_mse_all_outputs = df["MSE"].mean()
global_mae_all_outputs = df["MAE"].mean()

print(f'Global MSE per All Locations {global_mse_all_outputs}')
print(f'Global MAE per All Locations {global_mae_all_outputs}')
```
---

## 🧠 How It Works
### The Executor
1. Loads `TransformGraph` (`tft.TFTransformOutput`) and the SavedModel.
2. Builds an evaluation dataset via your **input function** (`input_fn_path`).
3. Computes **per-output** and **global** metrics across the dataset.
4. Writes a TFMA-like JSON to the `ModelEvaluation` artifact.

### Metrics Supported
- `mse`, `mae` (easy to extend in code).

## Output Format (metrics dictionary keys)
1. Per-output: `per_output>>{output_name}>>{metric}`
2. Global: `global>>{metric}`

---

## 📥 Dataset Function Contract
You supply the dataset function via `input_fn_path`, e.g. `"my_pkg.mdata:input_fn"`.

**Expected Signature**
```python
def input_fn(file_pattern: str, tf_transform_output, **kwargs) -> tf.data.Dataset:
    # returns a dataset of (features, labels)
```

**Expected Yields:**
1. `features`: a structure compatible with your model’s `SavedModel` signature.
2. `labels`: a Tensor shaped **(batch, 1, num_outputs)** so that:
    - `labels[:, 0, i]` is the true label vector for `output_names[i]`.
3. Your model’s predictions should have shape **(batch, num_outputs)** so that:
    - `predictions[:, i]` aligns with `output_names[i]`.

> If your labels are shaped differently, update either your input function or the executor’s slicing logic to align shapes.

---

## 📤 Artifact Output
- Writes a single JSON file under the `ModelEvaluation` artifact directory:
    - `evaluation.json`

- Example keys:
    - `metrics["per_output>>y_loc3>>mse"] = 0.0123`
    - `metrics["global>>mae"] = 0.1357`

---

## 📜 License
This project is licensed under the **MIT License**.  
© 2025 **Dr. Ahmed Moussa**

---

## 🤝 Contributing
Pull requests are welcome.  
For major changes, please open an issue first to discuss what you would like to change.

---

## 📫 Contact

For feedback, bugs, or collaboration ideas:

- **GitHub**: [@real-ahmed-moussa](https://github.com/real-ahmed-moussa)  

---

## ⭐️ Show Your Support

If you find this project useful, consider giving it a ⭐️ on [GitHub](https://github.com/real-ahmed-moussa//tfx-multioutput-evaluator)!