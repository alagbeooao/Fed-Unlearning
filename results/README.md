Per-run evaluation output goes here, one JSON per (seed, forgotten client):

    results_seed42_forget0.json
    results_seed42_forget2.json
    results_seed123_forget0.json
    results_seed123_forget2.json
    results_seed2024_forget0.json
    results_seed2024_forget2.json

`scripts/run_evaluation.py` writes them during a run. The three-seed results used in the paper are summarised in `aggregated_results.json`.
