import pandas as pd
from pathlib import Path

# ───────── chain score ─────────
def chain_score(row) -> float:
    """
    0.00  if intent wrong
    0.33  if only intent right
    0.66  if intent + tone right
    1.00  if all three right
    """
    if row["intent_pred"] != row["intent_true"]:
        return 0.0
    score = 1.0 / 3
    if row["tone_pred"] == row["tone_true"]:
        score += 1.0 / 3
        if row["resolution_pred"] == row["resolution_true"]:
            score += 1.0 / 3
    return round(score, 2)

# ───────── save helper ─────────
def save_chained_results(preds, data, df, out_dir: str, model_key: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"predictions_{model_key}.csv"

    y_test_df = data.y_test_df().reset_index(drop=True)
    texts     = df.reset_index(drop=True).iloc[data.X_test_idx]["text"].tolist()

    result_df = pd.DataFrame({
        "text"           : texts,
        "intent_true"    : y_test_df["intent"],
        "tone_true"      : [s.split(" | ")[1] for s in y_test_df["combo_23"]],
        "resolution_true": [s.split(" | ")[2] for s in y_test_df["combo_234"]],
        "intent_pred"    : preds["intent"],
        "tone_pred"      : [s.split(" | ")[1] for s in preds["combo_23"]],
        "resolution_pred": [s.split(" | ")[2] for s in preds["combo_234"]],
    })
    result_df["score"] = result_df.apply(chain_score, axis=1)
    result_df.to_csv(csv_path, index=False)
    print(f"📄 {csv_path} saved")
    return result_df
