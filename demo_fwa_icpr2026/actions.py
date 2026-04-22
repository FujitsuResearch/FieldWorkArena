# these are placeholders
# all these symbols will be available in browsergym actions
from typing import Literal


send_message_to_user: callable = None
report_infeasible_instructions: callable = None
retry_with_force: bool = False

"""IMPORTANT
The following primitives are meant to be included in the browsergym action using
inspect.getsource().
"""


def analyze_video(video_path: str, query: str):
    """
    Specifies the video file to analyze.

    Examples:
        analyze_video("sample.mp4", "How many people in this video?")
    """
    send_message_to_user(",".join(['analyze_video', video_path, query]))


def analyze_image(image_path: str, query: str):
    """
    Specifies the image file to analyze.

    Examples:
        analyze_image("sample.jpg", "How many people in this image?")
    """
    send_message_to_user(",".join(['analyze_image', image_path, query]))

