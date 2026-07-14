from argparse import ArgumentParser
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--type', '-t', type=str, default="huggingface") # huggingface or modelscope
    parser.add_argument('--name', '-n', type=str, default="MonkeyOCRv2-B-Parsing") # MonkeyOCRv2-S-Parsing
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model_weight", args.name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.type == "huggingface":
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="zenosai/"+args.name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
    elif args.type == "modelscope":
        from modelscope import snapshot_download
        snapshot_download(repo_id="zenosai/"+args.name, local_dir=model_dir)