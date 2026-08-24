import os
import subprocess
import re
import argparse
import sys
import json
from decimal import Decimal
from pathlib import Path

def format_threshold(value, precision):
    formatted = f"{value:.{precision}f}"
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def parse_threshold_from_dirname(dirname):
    if not dirname.startswith("th_"):
        return None
    try:
        return float(dirname[3:])
    except ValueError:
        return None


def extract_copypaste_metrics(text):
    pattern = r"copypaste:\s*([0-9]*\.?[0-9]+),\s*([0-9]*\.?[0-9]+),\s*([0-9]*\.?[0-9]+)"
    matches = re.findall(pattern, text)
    if not matches:
        return None
    precision, recall, hmean = matches[-1]
    return float(precision), float(recall), float(hmean)

def run_eval(th, config_file, num_gpus, dist_url, extra_args, base_search_dir, model_weights, test_dataset):
    # Create a unique output directory for this threshold inside base_search_dir
    output_dir = os.path.join(base_search_dir, f"th_{format_threshold(th, 3)}")

    # Construct the command
    cmd = [
        "python", "tools/train_net.py",
        "--config-file", config_file,
        "--eval-only",
        "--num-gpus", str(num_gpus),
        "--dist-url", dist_url,
        "OUTPUT_DIR", output_dir,
        "MODEL.WEIGHTS", model_weights,
        "MODEL.TRANSFORMER.INFERENCE_TH_TEST", str(th)
    ] + extra_args + ["DATASETS.TEST", f'("{test_dataset}",)']

    print(f"\n>>> Running evaluation with INFERENCE_TH_TEST = {format_threshold(th, 3)}")
    print(f">>> Using model weights: {model_weights}")
    print(f">>> Command: {' '.join(cmd)}")

    env = os.environ.copy()
    # Ensure CUDA_VISIBLE_DEVICES is set if not already set or override
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    # Run the command and capture output
    try:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        full_output = []
        for line in process.stdout:
            print(line, end="")
            full_output.append(line)
        process.wait()
        return "".join(full_output), output_dir
    except Exception as e:
        print(f"Error running command: {e}")
        return "", output_dir

def parse_f_score(output):
    metrics = extract_copypaste_metrics(output)
    if metrics is not None:
        return metrics[2]

    patterns = [
        r"DETECTION_ONLY_RESULTS.*hmean:\s+([\d\.]+)",
        r"DETECTION_ONLY_RESULTS.*'hmean':\s+([\d\.]+)",
        r"hmean:\s+([\d\.]+)", # Fallback
        r"'f_score':\s+([\d\.]+)" # Fallback if it's named f_score
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, output)
        if matches:
            # Take the last match as it's usually the final result
            return float(matches[-1])
            
    return None
def load_existing_results(search_results_dir):
    results = []
    search_path = Path(search_results_dir)
    if not search_path.is_dir():
        return results

    threshold_dirs = []
    for child in search_path.iterdir():
        if child.is_dir():
            th_value = parse_threshold_from_dirname(child.name)
            if th_value is not None:
                threshold_dirs.append((th_value, child))

    threshold_dirs.sort(key=lambda item: item[0])

    for th, th_dir in threshold_dirs:
        log_candidates = [th_dir / "log.txt", th_dir / "log.txt.rank0"]
        log_text = None
        for log_file in log_candidates:
            if log_file.is_file():
                log_text = log_file.read_text(errors="ignore")
                break

        if log_text is None:
            continue

        metrics = extract_copypaste_metrics(log_text)
        if metrics is None:
            continue

        results.append((th, metrics[2], str(th_dir)))

    return results


def infer_test_dataset_name(output_dir, requested_test_dataset):
    if requested_test_dataset:
        return requested_test_dataset

    search_root = Path(output_dir) / "threshold_search"
    if not search_root.is_dir():
        return "totaltext_poly_test"

    dataset_dirs = [child.name for child in search_root.iterdir() if child.is_dir()]
    if len(dataset_dirs) == 1:
        return dataset_dirs[0]

    return "totaltext_poly_test"

def plot_results(results, output_dir):
    """Plot the threshold search results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib not installed. Skipping plot generation.")
        print("Install it with: pip install matplotlib")
        return
    
    if not results:
        print("No results to plot.")
        return

    thresholds = [r[0] for r in results]
    f_scores = [r[1] for r in results]
    
    # Create figure with better styling
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the F-score curve
    ax.plot(thresholds, f_scores, 'b-o', linewidth=2, markersize=6, label='F-score (hmean)')
    
    # Highlight the best threshold
    best_idx = np.argmax(f_scores)
    best_th = thresholds[best_idx]
    best_f = f_scores[best_idx]
    ax.plot(best_th, best_f, 'r*', markersize=15, label=f'Best (TH={best_th:.2f}, F={best_f:.4f})')
    
    # Add grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Inference Threshold', fontsize=12)
    ax.set_ylabel('F-score (hmean)', fontsize=12)
    ax.set_title('Inference Threshold Search Results', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.4f}'))
    
    # Add value labels on points
    for th, f in zip(thresholds, f_scores):
        ax.annotate(f'{f:.4f}', (th, f), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, "search_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Also save a JSON file with detailed results for future analysis
    json_path = os.path.join(output_dir, "search_results.json")
    results_dict = {
        "thresholds": thresholds,
        "f_scores": f_scores,
        "best_threshold": float(best_th),
        "best_f_score": float(best_f)
    }
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results JSON saved to: {json_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Search for best INFERENCE_TH_TEST threshold by loading model from output directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python tools/search_th.py --output-dir output/r_50_poly/totaltext/finetune --start 0.3 --end 0.5 --step 0.01 --num-gpus 4
        """
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True,
        help="Path to the output directory containing config.yaml and model_final.pth"
    )
    parser.add_argument("--start", type=float, default=0.3, help="Start threshold")
    parser.add_argument("--end", type=float, default=0.5, help="End threshold")
    parser.add_argument("--step", type=float, default=0.005, help="Threshold step size")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:54325", help="Distributed training URL")
    parser.add_argument(
        "--test-dataset",
        type=str,
        default=None,
        help='Override DATASETS.TEST for evaluation, for example "inversetext_test" or "totaltext_poly_test_rotate"',
    )
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing threshold_search logs")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate plot of results")
    args, extra_args = parser.parse_known_args()

    # Validate and resolve the output directory
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f"ERROR: Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    # Check for config.yaml
    config_file = os.path.join(output_dir, "config.yaml")
    if not os.path.isfile(config_file):
        print(f"ERROR: config.yaml not found in {output_dir}")
        print("Make sure you have trained a model and it contains config.yaml")
        sys.exit(1)
    
    # Check for model_final.pth
    model_weights = os.path.join(output_dir, "model_final.pth")
    if not os.path.isfile(model_weights):
        print(f"ERROR: model_final.pth not found in {output_dir}")
        print("Make sure you have trained a model and it contains model_final.pth")
        sys.exit(1)

    test_dataset = infer_test_dataset_name(output_dir, args.test_dataset)
    test_dataset_dirname = test_dataset
    
    print(f"\n" + "="*60)
    print("  Threshold Search Configuration")
    print("="*60)
    print(f"Output Directory: {output_dir}")
    print(f"Test Dataset: {test_dataset}")
    print(f"Config File: {config_file}")
    print(f"Model Weights: {model_weights}")
    print(f"Search Range: [{args.start}, {args.end}] with step {args.step}")
    print(f"Number of GPUs: {args.num_gpus}")
    print("="*60 + "\n")

    step_decimal = Decimal(str(args.step))
    precision = max(-step_decimal.as_tuple().exponent, 0)

    # Create search results directory
    search_results_dir = os.path.join(output_dir, "threshold_search", test_dataset_dirname)
    os.makedirs(search_results_dir, exist_ok=True)

    if args.plot_only:
        results = load_existing_results(search_results_dir)
        if not results:
            print(f"\nNo existing threshold results found under: {search_results_dir}")
            print("Please confirm the threshold_search logs are already generated.")
            return

        best_row = max(results, key=lambda row: row[1])
        best_th = best_row[0]
        best_f = best_row[1]
    else:
        # Create the search space
        thresholds = []
        start_decimal = Decimal(str(args.start))
        end_decimal = Decimal(str(args.end))

        curr = start_decimal
        while curr <= end_decimal + Decimal("1e-12"):
            thresholds.append(float(curr))
            curr += step_decimal

        print(f"Searching thresholds: {[format_threshold(th, precision) for th in thresholds]}")

        best_f = -1.0
        best_th = -1.0
        results = []

        for th in thresholds:
            output, out_dir = run_eval(
                th,
                config_file,
                args.num_gpus,
                args.dist_url,
                extra_args,
                search_results_dir,
                model_weights,
                test_dataset,
            )
            f_score = parse_f_score(output)

            if f_score is not None:
                print(f"\n[Result] TH={format_threshold(th, precision)} -> F-score (hmean) = {f_score:.4f}")
                results.append((th, f_score, out_dir))
                if f_score > best_f:
                    best_f = f_score
                    best_th = th
            else:
                print(f"\n[Error] Could not parse F-score for TH={format_threshold(th, precision)}")

    if not results:
        print("\nNo results obtained. Please check the logs above.")
        return

    if args.plot_only:
        print(f"Loaded {len(results)} existing threshold results from: {search_results_dir}")

    print("\n" + "="*60)
    print("      Search Results Summary")
    print("="*60)
    print(f"{'Threshold':<12} | {'F-score':<10} | {'Output Dir'}")
    print("-" * 60)
    for th, f, out in results:
        marker = " <-- BEST" if th == best_th else ""
        print(f"{format_threshold(th, precision):<12} | {f:<10.4f} | {out}{marker}")
    
    print("-" * 60)
    print(f"BEST TH: {format_threshold(best_th, precision)} | BEST F-score: {best_f:.4f}")
    print("="*60)

    # Save summary to a file
    summary_path = os.path.join(search_results_dir, "search_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Search Results Summary:\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Test Dataset: {test_dataset}\n")
        f.write(f"Model Weights: {model_weights}\n")
        f.write(f"Search Range: [{args.start}, {args.end}] with step {args.step}\n")
        if args.plot_only:
            f.write("Mode: plot-only (loaded existing threshold_search logs)\n")
        f.write("\n")
        f.write(f"{'Threshold':<12} | {'F-score':<10} | {'Output Dir'}\n")
        f.write("-" * 60 + "\n")
        for th, fs, out in results:
            marker = " <-- BEST" if th == best_th else ""
            f.write(f"{format_threshold(th, precision):<12} | {fs:<10.4f} | {out}{marker}\n")
        f.write("-" * 60 + "\n")
        f.write(f"BEST TH: {format_threshold(best_th, precision)} | BEST F-score: {best_f:.4f}\n")
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Generate plot if requested
    if args.plot:
        plot_results(results, search_results_dir)
    
    print("\n" + "="*60)
    print("Threshold search completed!")
    print(f"Results directory: {search_results_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
