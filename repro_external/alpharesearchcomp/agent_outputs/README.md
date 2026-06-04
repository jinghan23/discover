# Public AlphaResearch Agent Outputs

This directory is for outputs that are actually shown as AlphaResearch agent
code/proposals in the public paper, not for post-hoc reproduction wrappers.

The public repository does not include the full AlphaResearch trajectories or
the final `best_program.py` files for each benchmark. The clearest published
"Research ideas -> Code implementation of the above idea" examples I found are
the two Third Autocorrelation checkpoints in Appendix E of the OpenReview PDF:

- `third_autocorrelation_e436c26a/agent_code.py`
- `third_autocorrelation_4f4c7847/agent_code.py`

Both are intentionally preserved with the missing `normalize_population`
reference, because that is part of the published failure analysis. Running them
answers whether that code implementation can discover the reported result.
