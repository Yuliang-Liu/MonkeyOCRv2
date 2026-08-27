import argparse
import random
import base64
import json
import urllib.request
from io import BytesIO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MonkeyOCRv2 understanding inference on a single image."
    )
    parser.add_argument(
        "--model-path",
        "-m",
        default="../model_weight/MonkeyOCRv2-S-Und",
        help="Path to the local MonkeyOCRv2 understanding model directory.",
    )
    parser.add_argument(
        "--server-url", "-s", default=None,
        help="vLLM OpenAI-compatible server URL. If omitted, run locally with Transformers.",
    )
    parser.add_argument("--served-model-name", default="MonkeyOCRv2")
    parser.add_argument(
        "--image-path",
        "-i",
        default="../images_test/ar.JPEG",
        help="Path or URL of the input image.",
    )
    parser.add_argument(
        "--question",
        "-q",
        default="What is the structure of the document in the image?",
        help="Question or instruction to ask about the input image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for Python and PyTorch generation.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1003520,
        help="Maximum number of pixels allowed for the input image before resizing.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature. Use 0 or a negative value for greedy decoding.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device used to run inference.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype used when loading the model.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation to use for inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)

    if args.server_url:
        image = args.image_path
        if not (image.startswith("http://") or image.startswith("https://") or image.startswith("data:")):
            # The OpenAI-compatible API has no max-pixels parameter. Resize
            # locally so vLLM receives the same image scale as Transformers.
            from PIL import Image, ImageOps

            pil_image = ImageOps.exif_transpose(Image.open(image)).convert("RGB")
            if args.max_pixels and pil_image.width * pil_image.height > args.max_pixels:
                scale = (args.max_pixels / float(pil_image.width * pil_image.height)) ** 0.5
                size = (max(1, int(pil_image.width * scale)), max(1, int(pil_image.height * scale)))
                pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG", quality=95)
            image = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        url = args.server_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        payload = {"model": args.served_model_name, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": args.question},
        ]}], "max_tokens": args.max_new_tokens, "temperature": max(args.temperature, 0)}
        request = urllib.request.Request(url + "/chat/completions", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
        print(result["choices"][0]["message"]["content"])
        return

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForCausalLM, AutoProcessor

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=args.dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": args.image_path,
                    "max_pixels": args.max_pixels,
                },
                {"type": "text", "text": args.question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = processor.decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )

    print(generated_text)


if __name__ == "__main__":
    main()
