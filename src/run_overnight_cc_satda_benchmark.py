#!/usr/bin/env python3
# ==============================================================================
# OVERNIGHT BENCHMARK ORCHESTRATOR FOR CC-SATDA V2
# Executes:
#   1. Loss Weight Sweeps (w_src, w_tgt) across N in {full, 200, 100}
#   2. Structural Ablations (v1 vs v2-EMA vs v2-EMA+Reliability)
#   3. Multi-Seed Averaging (5 seeds: 42, 123, 456, 789, 1011) across N in {full, 200, 100, 50}
#   4. Validation Tau Grid-Search per N
#   5. Aggregation into master JSON, CSV, and Markdown publication tables
# ==============================================================================

import argparse
import datetime
import json
import os
import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add directories to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_ultra_low_resource_study import (
    CLASSES, CLASS_TO_IDX,
    InMemorySkinDataset, build_transforms_b2_enhanced, set_seed
)
from train_cc_satda_v2 import (
    build_datasets_v2, train_cc_satda_v2_single, compute_class_priors
)


def run_benchmark(device_str: str = "cuda:3", data_dir: str = os.environ.get("CCSATDA_DATA", "data"),
                  out_dir: str = "outputs/cc_satda_v2", epochs: int = 15, steps_per_epoch: int = 60):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    start_time_all = time.time()
    print("================================================================================", flush=True)
    print(f"STARTING CC-SATDA V2 OVERNIGHT BENCHMARK SUITE ON {device}", flush=True)
    print(f"Output Directory: {out_dir}", flush=True)
    print(f"Start Timestamp:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Hyperparameters:  Epochs={epochs} | Steps/Epoch={steps_per_epoch}", flush=True)
    print("================================================================================\n", flush=True)

    all_results = []
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    def save_intermediate_records():
        csv_path = os.path.join(out_dir, "MASTER_V2_RESULTS.csv")
        json_path = os.path.join(out_dir, "MASTER_V2_RESULTS.json")
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # --------------------------------------------------------------------------
    # SUITE 1: LOSS WEIGHT SWEEPS (w_src, w_tgt) across N in {full, 200, 100}
    # --------------------------------------------------------------------------
    print("\n>>> SUITE 1: LOSS WEIGHT GRID SEARCH (w_src, w_tgt) <<<", flush=True)
    weight_pairs = [
        (1.0, 1.0, "symmetric_1.0_1.0"),
        (0.7, 1.3, "moderate_tgt_0.7_1.3"),
        (0.5, 1.5, "default_strong_tgt_0.5_1.5"),
        (0.3, 1.7, "extreme_tgt_0.3_1.7"),
    ]
    subsets_suite1 = ["full", 200, 100]

    for n in subsets_suite1:
        set_seed(42)
        datasets = build_datasets_v2(data_dir, seed=42)
        _, eval_tf_b2 = build_transforms_b2_enhanced(288)

        val_ds = InMemorySkinDataset(datasets["tgt_val"], eval_tf_b2, 1, img_size=288, cache_name="tgt_val")
        test_ds = InMemorySkinDataset(datasets["tgt_test"], eval_tf_b2, 1, img_size=288, cache_name="tgt_test")
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

        tgt_key = "tgt_train_full" if n == "full" else f"tgt_train_{n}"
        tgt_items = datasets[tgt_key]

        for w_s, w_t, desc in weight_pairs:
            tag = f"WeightSweep_N={n}_{desc}"
            print(f"\n--- Running {tag} (w_src={w_s}, w_tgt={w_t}) ---", flush=True)
            ckpt = os.path.join(ckpt_dir, f"best_weights_{n}_{desc}.pt")
            try:
                res = train_cc_satda_v2_single(
                    datasets["src_train"], tgt_items, val_loader, test_loader, device,
                    epochs=epochs, steps_per_epoch=steps_per_epoch,
                    w_source=w_s, w_target=w_t,
                    use_ema_prototypes=True,
                    use_reliability_weighting=True,
                    out_ckpt=ckpt, tag=tag
                )
                m_opt = res["test_metrics_optimal_tau"]
                m_raw = res["test_metrics_raw_tau0"]
                record = {
                    "suite": "1_weight_sweep",
                    "setting_N": str(n),
                    "target_samples": len(tgt_items),
                    "source_samples": len(datasets["src_train"]),
                    "seed": 42,
                    "w_source": w_s,
                    "w_target": w_t,
                    "weight_description": desc,
                    "ema_prototypes": True,
                    "reliability_weighting": True,
                    "optimal_tau": res["optimal_tau"],
                    "val_f1_optimal_tau": res["val_best_f1"],
                    "test_acc_optimal_tau": m_opt["accuracy"],
                    "test_f1_optimal_tau": m_opt["f1_macro"],
                    "test_auroc_optimal_tau": m_opt["auroc_macro_ovr"],
                    "test_mel_recall_optimal_tau": m_opt["melanoma_recall"],
                    "test_mel_f1_optimal_tau": m_opt["melanoma_f1"],
                    "test_bcc_recall_optimal_tau": m_opt["bcc_recall"],
                    "test_acc_raw_tau0": m_raw["accuracy"],
                    "test_f1_raw_tau0": m_raw["f1_macro"],
                    "test_mel_recall_raw_tau0": m_raw["melanoma_recall"],
                    "priors_source": res["priors"]["source"],
                    "priors_target": res["priors"]["target"],
                    "test_sweep_records": res["test_sweep_records"]
                }
                all_results.append(record)
                save_intermediate_records()
            except Exception as e:
                print(f"Error in {tag}: {e}", flush=True)

    # --------------------------------------------------------------------------
    # SUITE 2: ABLATION STUDY (v1 batch-only vs v2-EMA vs v2-EMA+Reliability)
    # --------------------------------------------------------------------------
    print("\n>>> SUITE 2: STRUCTURAL ABLATION STUDY <<<", flush=True)
    ablation_configs = [
        ("CC-SATDA_v1_batch_only", False, False),
        ("CC-SATDA_v2_EMA_only", True, False),
        ("CC-SATDA_v2_EMA_plus_Reliability", True, True),
    ]
    subsets_suite2 = [200, 100]

    for n in subsets_suite2:
        set_seed(42)
        datasets = build_datasets_v2(data_dir, seed=42)
        _, eval_tf_b2 = build_transforms_b2_enhanced(288)

        val_ds = InMemorySkinDataset(datasets["tgt_val"], eval_tf_b2, 1, img_size=288, cache_name="tgt_val")
        test_ds = InMemorySkinDataset(datasets["tgt_test"], eval_tf_b2, 1, img_size=288, cache_name="tgt_test")
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

        tgt_key = f"tgt_train_{n}"
        tgt_items = datasets[tgt_key]

        for method_name, use_ema, use_rel in ablation_configs:
            tag = f"Ablation_N={n}_{method_name}"
            print(f"\n--- Running {tag} ---", flush=True)
            ckpt = os.path.join(ckpt_dir, f"best_ablation_{n}_{method_name}.pt")
            try:
                res = train_cc_satda_v2_single(
                    datasets["src_train"], tgt_items, val_loader, test_loader, device,
                    epochs=epochs, steps_per_epoch=steps_per_epoch,
                    w_source=0.5, w_target=1.5,
                    use_ema_prototypes=use_ema,
                    use_reliability_weighting=use_rel,
                    out_ckpt=ckpt, tag=tag
                )
                m_opt = res["test_metrics_optimal_tau"]
                m_raw = res["test_metrics_raw_tau0"]
                record = {
                    "suite": "2_ablation_study",
                    "setting_N": str(n),
                    "target_samples": len(tgt_items),
                    "source_samples": len(datasets["src_train"]),
                    "seed": 42,
                    "w_source": 0.5,
                    "w_target": 1.5,
                    "method_variant": method_name,
                    "ema_prototypes": use_ema,
                    "reliability_weighting": use_rel,
                    "optimal_tau": res["optimal_tau"],
                    "val_f1_optimal_tau": res["val_best_f1"],
                    "test_acc_optimal_tau": m_opt["accuracy"],
                    "test_f1_optimal_tau": m_opt["f1_macro"],
                    "test_auroc_optimal_tau": m_opt["auroc_macro_ovr"],
                    "test_mel_recall_optimal_tau": m_opt["melanoma_recall"],
                    "test_mel_f1_optimal_tau": m_opt["melanoma_f1"],
                    "test_bcc_recall_optimal_tau": m_opt["bcc_recall"],
                    "test_acc_raw_tau0": m_raw["accuracy"],
                    "test_f1_raw_tau0": m_raw["f1_macro"],
                    "test_mel_recall_raw_tau0": m_raw["melanoma_recall"],
                    "priors_source": res["priors"]["source"],
                    "priors_target": res["priors"]["target"],
                    "test_sweep_records": res["test_sweep_records"]
                }
                all_results.append(record)
                save_intermediate_records()
            except Exception as e:
                print(f"Error in {tag}: {e}", flush=True)

    # --------------------------------------------------------------------------
    # SUITE 3: MULTI-SEED RIGOROUS BENCHMARK (5 SEEDS: 42, 123, 456, 789, 1011)
    # --------------------------------------------------------------------------
    print("\n>>> SUITE 3: MULTI-SEED RIGOROUS BENCHMARK <<<", flush=True)
    seeds = [42, 123, 456, 789, 1011]
    subsets_suite3 = ["full", 200, 100, 50]

    for n in subsets_suite3:
        for s in seeds:
            tag = f"MultiSeed_N={n}_Seed={s}"
            print(f"\n--- Running {tag} ---", flush=True)
            set_seed(s)
            datasets = build_datasets_v2(data_dir, seed=s)
            _, eval_tf_b2 = build_transforms_b2_enhanced(288)

            val_ds = InMemorySkinDataset(datasets["tgt_val"], eval_tf_b2, 1, img_size=288, cache_name="tgt_val")
            test_ds = InMemorySkinDataset(datasets["tgt_test"], eval_tf_b2, 1, img_size=288, cache_name="tgt_test")
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

            tgt_key = "tgt_train_full" if n == "full" else f"tgt_train_{n}"
            tgt_items = datasets[tgt_key]
            ckpt = os.path.join(ckpt_dir, f"best_multiseed_{n}_seed{s}.pt")

            try:
                res = train_cc_satda_v2_single(
                    datasets["src_train"], tgt_items, val_loader, test_loader, device,
                    epochs=epochs, steps_per_epoch=steps_per_epoch,
                    w_source=0.5, w_target=1.5,
                    use_ema_prototypes=True,
                    use_reliability_weighting=True,
                    out_ckpt=ckpt, tag=tag
                )
                m_opt = res["test_metrics_optimal_tau"]
                m_raw = res["test_metrics_raw_tau0"]
                record = {
                    "suite": "3_multiseed_benchmark",
                    "setting_N": str(n),
                    "target_samples": len(tgt_items),
                    "source_samples": len(datasets["src_train"]),
                    "seed": s,
                    "w_source": 0.5,
                    "w_target": 1.5,
                    "method_variant": "CC-SATDA_v2_Final",
                    "ema_prototypes": True,
                    "reliability_weighting": True,
                    "optimal_tau": res["optimal_tau"],
                    "val_f1_optimal_tau": res["val_best_f1"],
                    "test_acc_optimal_tau": m_opt["accuracy"],
                    "test_f1_optimal_tau": m_opt["f1_macro"],
                    "test_auroc_optimal_tau": m_opt["auroc_macro_ovr"],
                    "test_mel_recall_optimal_tau": m_opt["melanoma_recall"],
                    "test_mel_f1_optimal_tau": m_opt["melanoma_f1"],
                    "test_bcc_recall_optimal_tau": m_opt["bcc_recall"],
                    "test_acc_raw_tau0": m_raw["accuracy"],
                    "test_f1_raw_tau0": m_raw["f1_macro"],
                    "test_mel_recall_raw_tau0": m_raw["melanoma_recall"],
                    "priors_source": res["priors"]["source"],
                    "priors_target": res["priors"]["target"],
                    "test_sweep_records": res["test_sweep_records"]
                }
                all_results.append(record)
                save_intermediate_records()
            except Exception as e:
                print(f"Error in {tag}: {e}", flush=True)

    # --------------------------------------------------------------------------
    # 4. COMPUTE MULTI-SEED AGGREGATE SUMMARY (MEAN +- STD)
    # --------------------------------------------------------------------------
    print("\n========================================================", flush=True)
    print("COMPUTING MULTI-SEED AGGREGATE STATISTICS (MEAN +- STD)", flush=True)
    print("========================================================\n", flush=True)

    ms_records = [r for r in all_results if r["suite"] == "3_multiseed_benchmark"]
    grouped = defaultdict(list)
    for r in ms_records:
        grouped[r["setting_N"]].append(r)

    summary_rows = []
    for n_str in ["full", "200", "100", "50"]:
        if n_str not in grouped:
            continue
        items = grouped[n_str]
        accs = [x["test_acc_optimal_tau"] * 100 for x in items]
        f1s = [x["test_f1_optimal_tau"] for x in items]
        aurocs = [x["test_auroc_optimal_tau"] for x in items]
        mel_recs = [x["test_mel_recall_optimal_tau"] * 100 for x in items]
        bcc_recs = [x["test_bcc_recall_optimal_tau"] * 100 for x in items]
        mel_f1s = [x["test_mel_f1_optimal_tau"] for x in items]

        summary_rows.append({
            "Setting (N)": f"N={n_str}",
            "Runs": len(items),
            "Test Accuracy": f"{np.mean(accs):.2f}% ± {np.std(accs):.2f}%",
            "Macro F1": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
            "Macro AUROC": f"{np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}",
            "Melanoma Recall": f"{np.mean(mel_recs):.1f}% ± {np.std(mel_recs):.1f}%",
            "BCC Recall": f"{np.mean(bcc_recs):.1f}% ± {np.std(bcc_recs):.1f}%",
            "Melanoma F1": f"{np.mean(mel_f1s):.4f} ± {np.std(mel_f1s):.4f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, "AGGREGATE_V2_METRICS.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("MULTI-SEED AGGREGATE TABLE:")
    print(summary_df.to_string(index=False), flush=True)

    # Generate Markdown Report
    total_elapsed_min = (time.time() - start_time_all) / 60.0
    report_md_path = os.path.join(out_dir, "BENCHMARK_REPORT_V2.md")
    with open(report_md_path, "w") as f:
        f.write("# CC-SATDA v2: Comprehensive Benchmark & Ablation Report\n\n")
        f.write(f"- **Execution Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total Elapsed Time:** {total_elapsed_min:.1f} minutes\n")
        f.write(f"- **Total Benchmark Runs:** {len(all_results)}\n\n")
        f.write("## 1. Multi-Seed Headline Summary (Mean ± Std across Seeds)\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## 2. All Benchmark Records\n\n")
        df_all = pd.DataFrame(all_results)
        f.write(df_all[["suite", "setting_N", "seed", "optimal_tau", "test_acc_optimal_tau", "test_f1_optimal_tau", "test_auroc_optimal_tau", "test_mel_recall_optimal_tau"]].to_markdown(index=False))
        f.write("\n")

    print(f"\n[FINISHED] Report written to {report_md_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps-per-epoch", type=int, default=60)
    args = parser.parse_args()

    run_benchmark(device_str=args.device, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)
