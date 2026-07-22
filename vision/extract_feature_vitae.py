import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Extract image features with MonkeyOCRv2-AS.")
    parser.add_argument(
        "--model-path", "-m",
        default="../model_weight/MonkeyOCRv2-AS",
        help="Path or Hugging Face model name for the vision model.",
    )
    parser.add_argument(
        "--image-path", "-i",
        default="../images_test/ar.JPEG",
        help="Path to the input image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from transformers import AutoModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoImageProcessor
    from PIL import Image
    from einops import rearrange

    model_name = args.model_path
    img_path = args.image_path

    img_ori = Image.open(img_path).convert("RGB")
    model = AutoModel.from_pretrained(model_name, dtype="auto", device_map="auto", trust_remote_code=True) 

    image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True, max_pixels=1802240)

    messages = [
        {
            "content": [
                 {
                    "type": "image",
                    "image": img_path,
                },
            ],
        }
    ]
    image_inputs, _ = process_vision_info(messages, image_patch_size=image_processor.patch_size*image_processor.merge_size/2)

    media_inputs =  image_processor(images=image_inputs, videos=None, return_tensors='pt')

    pixel_values = media_inputs['pixel_values'].to(model.device)
    image_grid_thw = media_inputs['image_grid_thw'].to(model.device)

    # 4 stage vision embeddings, each stage has different grid_hw (1/4, 1/8, 1/16, 1/32)
    vision_embeddings_all, grid_hw_all = model(pixel_values, image_grid_thw)

    images_features = [rearrange(vision_embeddings , '(h w) c -> h w c ', h=grid_hw[0][0], w=grid_hw[0][1]) for vision_embeddings, grid_hw in zip(vision_embeddings_all, grid_hw_all)]
    print("Input image size:", img_ori.size, "Extracted feature shapes:", [feature.shape for feature in images_features])


if __name__ == "__main__":
    main()
