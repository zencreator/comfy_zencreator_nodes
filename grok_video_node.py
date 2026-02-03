from comfy_api.latest import IO
from comfy_api_nodes.util import download_url_to_video_output

from xai_sdk import Client


class GrokTextToVideoApi(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokTextToVideoApi",
            display_name="Grok Text to Video",
            category="api node/video/Grok",

            inputs=[
                IO.String.Input(
                    "api_key",
                    default="",
                    tooltip="xAI API key",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                ),
                IO.Combo.Input(
                    "duration",
                    options=["1","2","3","4","5","6","7","8","9","10",
                             "11","12","13","14","15"],
                    default="5",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9","4:3","1:1",
                             "9:16","3:4","3:2","2:3"],
                    default="16:9",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["720p","480p"],
                    default="720p",
                ),
            ],

            outputs=[
                IO.Video.Output(),
            ],

            hidden=[
                IO.Hidden.unique_id,
            ],

            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        api_key: str,
        prompt: str,
        duration: str,
        aspect_ratio: str,
        resolution: str,
    ):
        if not api_key:
            raise ValueError("API key required")

        client = Client(api_key=api_key)

        response = client.video.generate(
            prompt=prompt,
            model="grok-imagine-video",
            duration=int(duration),
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )

        return IO.NodeOutput(
            await download_url_to_video_output(response.url)
        )


# -------------------------------------------------
# REQUIRED by custom_nodes loader
# -------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "GrokTextToVideoApi": GrokTextToVideoApi,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokTextToVideoApi": "Grok Text to Video",
}
