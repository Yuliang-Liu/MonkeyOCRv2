from transformers import AutoModel
from qwen_vl_utils import process_vision_info
from transformers import AutoImageProcessor
from PIL import Image
from einops import rearrange

model_path = "../model_weight/MonkeyOCRv2-B"

img_path = "../images_test/ar.JPEG"
img_ori = Image.open(img_path).convert("RGB")
model = AutoModel.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True, max_pixels=1003520)

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
image_inputs, video_inputs = process_vision_info(messages)


media_inputs =  image_processor(images=image_inputs, videos=None, return_tensors='pt')

pixel_values = media_inputs['pixel_values'].to(model.device)
image_grid_thw = media_inputs['image_grid_thw'].to(model.device)

vision_embeddings = model(pixel_values, image_grid_thw)

h = image_grid_thw[0][1]//model.config.spatial_merge_size
w = image_grid_thw[0][2]//model.config.spatial_merge_size
m = model.config.spatial_merge_size

images_feature = rearrange(vision_embeddings , '(h w m1 m2) c -> (h m1) (w m2) c ',h=h, w=w, m1=m, m2=m)

print("Input image shape:", pixel_values.shape, "Extracted feature shape:", images_feature.shape)