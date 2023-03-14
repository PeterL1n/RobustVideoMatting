import torch
from model import MattingNetwork
from inference import convert_video

from cog import BasePredictor, Path, Input


class Predictor(BasePredictor):
    def setup(self):
        self.model = MattingNetwork('resnet50').eval().cuda()
        self.model.load_state_dict(torch.load('rvm_resnet50.pth'))

    def predict(
            self,
            input_video: Path = Input(description="Video to segment."),
            output_type: str = Input(default="green-screen", choices=["green-screen", "alpha-mask", "foreground-mask"]),

    ) -> Path:

        convert_video(
            self.model,  # The model, can be on any device (cpu or cuda).
            input_source=str(input_video),  # A video file or an image sequence directory.
            output_type='video',  # Choose "video" or "png_sequence"
            output_composition='green-screen.mp4',  # File path if video; directory path if png sequence.
            output_alpha="alpha-mask.mp4",  # [Optional] Output the raw alpha prediction.
            output_foreground="foreground-mask.mp4",  # [Optional] Output the raw foreground prediction.
            output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
            downsample_ratio=None,  # A hyperparameter to adjust or use None for auto.
            seq_chunk=12,  # Process n frames at once for better parallelism.
        )
        output_type = str(output_type)
        return Path(f'{output_type}.mp4')
