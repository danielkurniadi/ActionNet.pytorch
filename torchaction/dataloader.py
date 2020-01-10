import nvidia
import nvidia.dali.ops as ops
import nvidia.dali.types as dtypes
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class VideoPipeline(Pipeline):
    """ Pipeline for reading H264 videos based on NVIDIA DALI.
	Returns a batch of sequences of `sequence_length` frames of shape [N, F, C, H, W]
	(N being the batch size and F the number of frames). Frames are RGB uint8.
    
    Arguments:
    ------------------------------
        .. batch_size (int):
                size of batches
        .. sequence_length (int):
                length of video clip, the time dimension
        .. fileroot (str):
                video root directory which has been structured as standard labelled video dir
        .. crop_size (array-like):
        .. num_threads (int):
                number of threads spawned for loading video
        .. device_id (int):
                gpu device id to be used
        .. output_layout (str, optional)
        .. random_shuffle (int, optional)
        .. step (int, optional)
    """

    def __init__(self, batch_size, sequence_length, fileroot, crop_size, 
                num_threads, device_id, output_layout=dtypes.NCHW):
        super().__init__(batch_size, num_threads, device_id, seed=42)

        # Define video reader
        self.reader = ops.VideoReader(device="gpu",)
