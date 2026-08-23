import argparse
import sys
from pathlib import Path

PARSING_DIR = Path(__file__).resolve().parents[1]
if str(PARSING_DIR) not in sys.path:
    sys.path.insert(0, str(PARSING_DIR))

from cpu.core_runner import (
    BackendConfig,
    BackendManager,
    PipelineConfig,
    run_pipeline,
    run_single_task_recognition,
)


def main():
    parser = argparse.ArgumentParser(description="Run MonkeyOCRv2-B-Parsing locally on CPU.")
    parser.add_argument("--input-path", "-i", default="../images_test", help="Input file or folder containing PDFs, images, or both")
    parser.add_argument("--model-path", "-m", default="../model_weight/MonkeyOCRv2-B-Parsing", help="Model path")
    parser.add_argument("--output-path", "-o", default="./output/cpu_test", help="Output directory")
    parser.add_argument("--max-pixels", type=int, default=1003520, help="Maximum input image pixels; larger images are resized proportionally")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32", help="Torch dtype for CPU Transformers inference")
    parser.add_argument("--attn-implementation", default="eager", help="Attention implementation for CPU Transformers inference")
    parser.add_argument("--preprocess-batch-size", type=int, default=32, help="Preprocessor batch size")
    parser.add_argument("--page-max-inflight", type=int, default=1, help="Maximum pages kept/submitted concurrently during parsing")
    parser.add_argument("--draw-layout", action="store_true", help="Save layout visualization PDFs to output_path/layout")
    parser.add_argument("--end2end", action="store_true", help="Enable end-to-end parsing and output bbox/label/content from the full image")
    parser.add_argument("--skip-processed", action="store_true", help="Skip documents whose markdown output already exists")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing and directly parse the original input documents")
    parser.add_argument("-t", "--task", choices=["text", "formula", "table"], help="Single task recognition type")
    parser.add_argument("--retry-repeat", action="store_true", help="Retry recognition outputs when repeated tokens are detected")
    parser.add_argument("--retry-repeat-max-retries", type=int, default=3, help="Maximum retry attempts for repeated-token outputs")
    parser.add_argument("--keep-header-footer", action="store_true", help="Keep Page-header and Page-footer blocks in markdown output")
    parser.add_argument("--use-base64", "--use_base64", action="store_true", default=False, help="Write Picture blocks as base64 in markdown")
    args = parser.parse_args()

    backend_config = BackendConfig(
        model_path=args.model_path,
        backend="transformers",
        device="cpu",
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_pixels=args.max_pixels,
        preprocess_batch_size=args.preprocess_batch_size,
        skip_preprocess=args.skip_preprocess or bool(args.task),
    )
    backend_manager = BackendManager()

    try:
        if args.task:
            result = run_single_task_recognition(
                args.input_path,
                args.output_path,
                args.task,
                backend_config,
                backend_manager=backend_manager,
            )
            print(
                f"Single task ({args.task}) recognition completed in {result['elapsed']:.2f}s. "
                f"Results saved to {result['out_dir']}"
            )
            return

        run_pipeline(
            PipelineConfig(
                input_path=args.input_path,
                output_path=args.output_path,
                backend=backend_config,
                page_max_inflight=args.page_max_inflight,
                draw_layout=args.draw_layout,
                end2end=args.end2end,
                skip_processed=args.skip_processed,
                retry_repeat=args.retry_repeat,
                retry_repeat_max_retries=args.retry_repeat_max_retries,
                keep_header_footer=args.keep_header_footer,
                use_base64=args.use_base64,
                show_progress_bar=True,
            ),
            backend_manager=backend_manager,
        )
    finally:
        backend_manager.close()


if __name__ == "__main__":
    main()
