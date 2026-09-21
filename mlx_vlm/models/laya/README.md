# Laya decisions on MLX

The published `convaiinnovations/laya` checkpoint contains three ModernBERT decision models: English, `multilingual`, and `typed-decisions`. Load the original Hugging Face checkpoint directly; no converted repository or generic conversion command is required.

```python
from mlx_vlm.models.laya import load

model = load("convaiinnovations/laya", subfolder="typed-decisions")
result = model.predict(
    {"ticket": "Please refund my duplicate charge"},
    {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": None, "technical": None, "sales": None},
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer request a refund?",
        },
    },
)
print(result["answers"])
```

Use `load("convaiinnovations/laya")` for English and `subfolder="multilingual"` for multilingual inputs. Each question returns its constrained choice, score, or Boolean probability; this model does not generate text. The decision head and encoder are loaded strictly from the published weights. The API is separate from VLM chat and token generation.
