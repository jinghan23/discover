# Cap Set Priority Function

Reproduces the public FunSearch cap-set results from the official DeepMind repository:

- `n=8`: FunSearch priority function builds a valid cap set of size `512`.
- `n=8`: the human-readable explicit construction derived from that function also builds the same `512` points.
- `n=9`: FunSearch priority function builds a valid cap set of size `1082`.

Sources:

- https://github.com/google-deepmind/funsearch/blob/main/cap_set/cap_set.ipynb
- https://github.com/google-deepmind/funsearch
- https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/Mathematical-discoveries-from-program-search-with-large-language-models.pdf

Run:

```bash
python repro_external/cap_set_priority_function/reproduce_cap_set.py
```

The script writes `cap_set_results.json` in this folder.
