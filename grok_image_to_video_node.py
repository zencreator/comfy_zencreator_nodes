from comfy_api.latest import IO
from comfy_api_nodes.util import (
    tensor_to_base64_string,
    download_url_to_video_output,
    get_number_of_images,
)

from xai_sdk import Client


SUPPORTED_RATIOS = {
    "1:1": 1.0,
    "4:3": 4/3,
    "3:4": 3/4,
    "16:9": 16/9,
    "9:16": 9/16,
    "3:2": 3/2,
    "2:3": 2/3,
}


def closest_ratio(w, h):
    r = w / h
    return min(
        SUPPORTED_RATIOS.items(),
        key=lambda x: abs(x[1] - r)
    )[0]


class GrokImageToVideoApi(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokImageToVideoApi",
            display_name="Grok Image to Video",
            category="api node/video/Grok",

            inputs=[
                IO.String.Input("api_key", default=""),
                IO.Image.Input("image"),
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Combo.Input(
                    "duration",
                    options=["1","2","3","4","5","6","7","8","9","10",
                             "11","12","13","14","15"],
                    default="5",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["720p","480p"],
                    default="720p",
                ),
            ],

            outputs=[IO.Video.Output()],
            hidden=[IO.Hidden.unique_id],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        api_key,
        image,
        prompt,
        duration,
        resolution,
    ):
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one image required")

        # Detect image size
        tensor = image[0]
        h, w = tensor.shape[:2]

        aspect_ratio = closest_ratio(w, h)

        image_b64 = tensor_to_base64_string(
            image,
            total_pixels=4096 * 4096
        )

        image_data_url = "data:image/png;base64," + image_b64

        client = Client(api_key=api_key)

        response = client.video.generate(
            prompt=prompt,
            model="grok-imagine-video",
            image_url=image_data_url,
            duration=int(duration),
            resolution=resolution,
            aspect_ratio=aspect_ratio,
        )

        return IO.NodeOutput(
            await download_url_to_video_output(response.url)
        )


NODE_CLASS_MAPPINGS = {
    "GrokImageToVideoApi": GrokImageToVideoApi,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokImageToVideoApi": "Grok Image to Video",
}
