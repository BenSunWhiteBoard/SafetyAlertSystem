import pyaudio
import wave
import struct

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")   
# please manually install torch 1.9 if Colab changes its default version

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def monitorVideoStream(predictor, input_video_file = 'example_video.mp4'):

    # create a VideoCapture object and read from input file
    # pass 0 to use the camera as input
    cap = cv2.VideoCapture(input_video_file)

    if (cap.isOpened()== False): 
      print("Error opening video stream or file")

    # read video frame one by one
    while(cap.isOpened()):
      
      ret, frame = cap.read()
      if ret == True:

        # do object detection on the input video frame
        outputs = predictor(frame)
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2_imshow(out.get_image()[:, :, ::-1])

        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

      # Break the loop
      else: 
        break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # object detection model
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml")
    predictor = DefaultPredictor(cfg)

    CHANNELS = 1  # Number of channels
    RATE = 16000  # Sampling rate (frames/second)
    BLOCKLEN = 1024  # block length in samples
    DURATION = 10       # Duration in seconds
    LENGTH = DURATION * RATE  # Signal length
    WIDTH = 2  # Number of bytes per sample

    K = int(LENGTH / BLOCKLEN)  # Number of blocks

    # read from microphone
    p = pyaudio.PyAudio()
    stream = p.open(
        format      = p.get_format_from_width(WIDTH),
        channels    = 1,
        rate        = RATE,
        input       = True,
        output      = False
    )
    MAXVALUE = 2**15-1 
    threshold = MAXVALUE // 5
    
    # keep listening to sus sound
    for _ in range(K):
        
        input_bytes = stream.read(BLOCKLEN)

        input_block = struct.unpack("h" * BLOCKLEN, input_bytes)

        loud_frames = sum([2 if abs(input_block[i]) >= threshold else 0 for i in range(BLOCKLEN) ])
        
        something_happened = loud_frames > (BLOCKLEN / 10)
        
        if(something_happened):
            print("read video frame from the camera, press q to stop")
            # read video frame from the camera, press q to stop
            monitorVideoStream(0)

    stream.stop_stream()
    stream.close()
    p.terminate()