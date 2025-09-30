# ZenCreator - BytePlus Seedream 4.0 for ComfyUI

Custom node package for ComfyUI that integrates BytePlus Seedream 4.0 image generation API.

## Features

- **Multiple Models**: Support for seedream-4-0-250828 and ep-20250918135640-cxht8
- **Flexible Sizing**: Presets for 1K, 2K, 4K resolutions with various aspect ratios (1:1, 16:9, 9:16, 4:3, 3:4)
- **Custom Dimensions**: Support for custom width and height (512-4096px)
- **Sequential Generation**: Generate 1-15 images in a single request
- **Image Input**: Support for up to 5 input images or image URLs
- **Seed Control**: Reproducible results with seed values (for supported models)
- **Response Formats**: URL or base64 JSON responses
- **Watermark Control**: Optional watermark on generated images

## Installation

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes/
```

2. Clone or create the zencreator folder:
```bash
mkdir -p zencreator
cd zencreator
```

3. Add all the package files to this folder:
   - `__init__.py`
   - `BytePlusSeedreamNode.py`
   - `requirements.txt`
   - `README.md`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Restart ComfyUI

## Usage

1. Add the **BytePlus Seedream 4.0** node from the `ZenCreator/BytePlus` category
2. Enter your BytePlus API key
3. Configure the generation parameters:
   - **Model**: Choose between available models
   - **Prompt**: Describe the image you want to generate
   - **Size**: Select from preset sizes or choose "Custom" for custom dimensions
   - **Seed**: Use -1 for random or specify a seed for reproducibility
   - **Max Images**: Generate 1-15 images (when sequential generation is enabled)
4. Optionally connect input images for image-to-image generation

## Parameters

### Required

- **api_key**: Your BytePlus API key (password protected)
- **model**: Model selection (seedream-4-0-250828 or ep-20250918135640-cxht8)
- **prompt**: Text description of the desired image
- **size**: Image size preset (1K/2K/4K with various aspect ratios)
- **custom_width/height**: Used when "Custom" size is selected
- **seed_value**: Random seed (-1 for random, 0-2147483647 for specific seed)
- **sequential_image_generation**: Enable/disable batch generation
- **max_images**: Number of images to generate (1-15)
- **watermark**: Enable/disable watermark
- **response_format**: URL or base64 JSON

### Optional

- **image1-5**: Input images for image-to-image generation
- **image_url**: URL of an input image

## Output

- **images**: List of generated images as ComfyUI IMAGE tensors
- **usage_info**: API usage information and generation parameters

## Size Presets

### 1K Resolution
- Auto (1K)
- 1024x1024 (1:1)
- 1280x720 (16:9)
- 720x1280 (9:16)
- 1152x864 (4:3)
- 864x1152 (3:4)

### 2K Resolution
- Auto (2K)
- 2048x2048 (1:1)
- 2560x1440 (16:9)
- 1440x2560 (9:16)
- 2304x1728 (4:3)
- 1728x2304 (3:4)

### 4K Resolution
- Auto (4K)
- 4096x4096 (1:1)
- 3840x2160 (16:9)
- 2160x3840 (9:16)
- 4096x3072 (4:3)
- 3072x4096 (3:4)

## Notes

- The seedream-4-0-250828 model does not support seed values
- API requests have a 120-second timeout
- Image downloads have a 30-second timeout
- Input images are automatically converted to base64 PNG format

## Requirements

- Python 3.7+
- torch
- requests
- pillow
- numpy

## License

[Your License Here]

## Support

For issues and questions, please refer to the BytePlus API documentation or create an issue in the repository.