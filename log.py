import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPLITS = ["clean", "seen", "holdout"]


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_summary(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_bundle(folder: Path):
    data = {
        "summary": load_summary(folder / "summary.json"),
        "episodes": {}
    }
    for split in SPLITS:
        data["episodes"][split] = load_jsonl(folder / f"{split}_episodes.jsonl")
    return data


def get_metric(summary, split, key):
    return summary.get(split, {}).get(key, np.nan)


def make_outdir(base: Path):
    outdir = base / "plots"
    outdir.mkdir(exist_ok=True)
    return outdir


def save_table(model_data, outdir: Path):
    rows = []
    for model_name, bundle in model_data.items():
        summary = bundle["summary"]
        for split in SPLITS:
            rows.append({
                "model": model_name,
                "split": split,
                "episode_score_mean": get_metric(summary, split, "episode_score_mean"),
                "clean_score_mean": get_metric(summary, split, "clean_score_mean"),
                "fault_score_mean": get_metric(summary, split, "fault_score_mean"),
                "episode_detection_rate": get_metric(summary, split, "episode_detection_rate"),
                "clean_false_alarm_episode_rate": get_metric(summary, split, "clean_false_alarm_episode_rate"),
                "step_precision": get_metric(summary, split, "step_precision"),
                "step_recall": get_metric(summary, split, "step_recall"),
                "step_f1": get_metric(summary, split, "step_f1"),
                "episode_unique_states_mean": get_metric(summary, split, "episode_unique_states_mean"),
                "episode_revisit_ratio_mean": get_metric(summary, split, "episode_revisit_ratio_mean"),
                "recent_novel_rate_mean": get_metric(summary, split, "recent_novel_rate_mean"),
                "unique_action_bigrams_mean": get_metric(summary, split, "unique_action_bigrams_mean"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "summary_compare_table.csv", index=False, encoding="utf-8-sig")
    print(f"saved: {outdir / 'summary_compare_table.csv'}")


def grouped_bar(model_data, split, metric_keys, title, out_path):
    model_names = list(model_data.keys())
    x = np.arange(len(metric_keys))
    width = 0.8 / len(model_names)

    plt.figure(figsize=(12, 5))
    for i, model_name in enumerate(model_names):
        vals = [get_metric(model_data[model_name]["summary"], split, k) for k in metric_keys]
        plt.bar(x + (i - (len(model_names) - 1) / 2) * width, vals, width=width, label=model_name)

    plt.xticks(x, metric_keys, rotation=20, ha="right")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved: {out_path}")


def plot_score_box(model_data, split, out_path):
    vals = []
    labels = []
    for model_name, bundle in model_data.items():
        df = bundle["episodes"].get(split, pd.DataFrame())
        if not df.empty and "episode_score" in df.columns:
            scores = pd.to_numeric(df["episode_score"], errors="coerce").dropna().values
            if len(scores) > 0:
                vals.append(scores)
                labels.append(model_name)

    if not vals:
        return

    plt.figure(figsize=(9, 5))
    plt.boxplot(vals, labels=labels, showfliers=False)
    plt.ylabel("episode_score")
    plt.title(f"Episode score distribution ({split})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved: {out_path}")


def plot_score_ma(model_data, split, out_path, window=20):
    plt.figure(figsize=(10, 5))
    plotted = False

    for model_name, bundle in model_data.items():
        df = bundle["episodes"].get(split, pd.DataFrame())
        if df.empty or "episode_score" not in df.columns:
            continue
        scores = pd.to_numeric(df["episode_score"], errors="coerce").dropna()
        if len(scores) == 0:
            continue
        ma = scores.rolling(window=window, min_periods=1).mean()
        x = np.arange(1, len(ma) + 1)
        plt.plot(x, ma, label=model_name)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("episode index")
    plt.ylabel("episode_score")
    plt.title(f"Episode score moving average ({split})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved: {out_path}")


def main():
    base = Path(__file__).resolve().parent

    folders = {
        "baseline": base / "baseline",
        "v13": base / "v13",
        "v16": base / "v16",
    }

    for name, folder in folders.items():
        if not folder.exists():
            raise FileNotFoundError(f"{name} folder not found: {folder}")
        if not (folder / "summary.json").exists():
            raise FileNotFoundError(f"summary.json not found in: {folder}")

    model_data = {name: load_bundle(folder) for name, folder in folders.items()}
    outdir = make_outdir(base)

    save_table(model_data, outdir)

    for split in SPLITS:
        grouped_bar(
            model_data,
            split,
            ["episode_score_mean", "clean_score_mean", "fault_score_mean"],
            f"Score metrics ({split})",
            outdir / f"01_score_metrics_{split}.png",
        )

        grouped_bar(
            model_data,
            split,
            [
                "episode_detection_rate",
                "clean_false_alarm_episode_rate",
                "step_precision",
                "step_recall",
                "step_f1",
            ],
            f"Detection metrics ({split})",
            outdir / f"02_detection_metrics_{split}.png",
        )

        grouped_bar(
            model_data,
            split,
            [
                "episode_unique_states_mean",
                "episode_revisit_ratio_mean",
                "recent_novel_rate_mean",
                "unique_action_bigrams_mean",
            ],
            f"Coverage metrics ({split})",
            outdir / f"03_coverage_metrics_{split}.png",
        )

        plot_score_box(
            model_data,
            split,
            outdir / f"04_episode_score_box_{split}.png",
        )

        plot_score_ma(
            model_data,
            split,
            outdir / f"05_episode_score_ma_{split}.png",
            window=20,
        )

    print(f"\n완료: {outdir}")


if __name__ == "__main__":
    main()
