#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import cv2
from tqdm import tqdm

from projectaria_tools.core import data_provider,calibration, mps
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.vrs import extract_audio_track
from projectaria_tools.core.mps.utils import get_nearest_eye_gaze, get_gaze_vector_reprojection


def extract_audio(vrs_file_path: str) -> str:
    """Extract audio from a VRS file as a wav file in a temporary folder."""
    temp_folder = tempfile.mkdtemp()
    if not temp_folder:
        return None
    # else continue process vrs audio extraction
    json_output_string = extract_audio_track(
        vrs_file_path, os.path.join(temp_folder, "audio.wav")
    )
    json_output = json.loads(json_output_string)  # Convert string to Dict
    if json_output and json_output["status"] == "success":
        return json_output["output"]
    # Else we were not able to export a Wav file from the VRS file
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vrs",
        type=str,
        required=True,
        help="path to the VRS file to be converted to a video",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="path where to store the output frames and gaze file",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        required=False,
        default=1,
        help="Downsampling factor on VRS images (Must be >=1)",
    )
    parser.add_argument(
        "--eyegaze",
        type=str,
        help="path to the MPS eye gaze file",
    )    
    return parser.parse_args()


class Vrs2FramesPyFrameConverter:
    """
    Class used to convert the VRS RGB frame to a sequence of frames.
    A Vrs2FramesPyFrameConverter object is defined as callable to be used as a make_frame(t) functor by moviepy.
    """

    def sample_frame_and_timestamp(self, image_data_and_record) -> [np.ndarray, int]:
        """
        Return the image frame and corresponding timestamp.
            Image is down sampled and rotated if required.
        """
        img = image_data_and_record.image_data_and_record()[0].to_numpy_array().copy()

        if self.down_sampling_factor_ > 1:
            img = img[:: self.down_sampling_factor_, :: self.down_sampling_factor_]
        # Rotate image
        img = np.rot90(img, -1)

        capture_timestamp = image_data_and_record.image_data_and_record()[
            1
        ].capture_timestamp_ns
        return [img, capture_timestamp]

    def __init__(self, vrs_path: str, eye_gaze: str = None, down_sampling_factor: int = 1, output_folder: str = None):
        self.down_sampling_factor_ = down_sampling_factor
        self.output_folder_ = output_folder

        self.eyegaze_data = mps.read_eyegaze(eye_gaze) if eye_gaze else None
        ##
        # Initialize the VRS data provider
        self.provider_ = data_provider.create_vrs_data_provider(vrs_path)
        if not self.provider_:
            raise ValueError(f"vrs file: '{vrs_path}' cannot be read")

        self.rgb_stream_id_ = StreamId("214-1")

        if self.eyegaze_data:
            self.rgb_stream_label_ = self.provider_.get_label_from_stream_id(self.rgb_stream_id_)
            self.device_calibration_ = self.provider_.get_device_calibration()
            self.T_device_CPF_ = self.device_calibration_.get_transform_device_cpf()
            self.rgb_camera_calibration_ = self.device_calibration_.get_camera_calib(self.rgb_stream_label_)
            
            rgb_linear_camera_calibration = calibration.get_linear_camera_calibration(
                int(self.rgb_camera_calibration_.get_image_size()[0]),
                int(self.rgb_camera_calibration_.get_image_size()[1]),
                self.rgb_camera_calibration_.get_focal_lengths()[0],
                "pinhole",
                self.rgb_camera_calibration_.get_transform_device_camera(),
            )
            self.rgb_camera_calibration_ = calibration.rotate_camera_calib_cw90deg(
                rgb_linear_camera_calibration
            )
        ##
        # Configure a deliver queue to provide only RGB image data stream

        deliver_option = self.provider_.get_default_deliver_queued_options()
        deliver_option.deactivate_stream_all()
        deliver_option.activate_stream(self.rgb_stream_id_)

        self.seq_ = self.provider_.deliver_queued_sensor_data(deliver_option)
        self.iter_data_ = iter(self.seq_)
        image_data_and_record = next(self.iter_data_)
        self.last_valid_frame_, self.last_timestamp_ = self.sample_frame_and_timestamp(
            image_data_and_record
        )
        self.first_timestamp_ = self.last_timestamp_
        self.dropped_frames_count_ = 0
        self.frame_count = 0

    def log_eye_gaze(self, device_time_ns: int) -> np.ndarray:
        if self.eyegaze_data:
            eye_gaze = get_nearest_eye_gaze(self.eyegaze_data, device_time_ns)
            if eye_gaze:
                # If depth available use it, else use a proxy (1 meter depth along the EyeGaze ray)
                depth_m = eye_gaze.depth or 1.0

                # Compute eye_gaze vector at depth_m reprojection in the image
                gaze_projection = get_gaze_vector_reprojection(
                    eye_gaze,
                    self.rgb_stream_label_,
                    self.device_calibration_,
                    self.rgb_camera_calibration_,
                    depth_m,
                )
                if gaze_projection is not None:
                    return gaze_projection / self.down_sampling_factor_;
                # Else (eye gaze projection is outside the image or behind the image plane)
        return None

    def export_frames(self) -> np.ndarray:
        """
        Create a functor compatible with the make_frame(t) functor concept of moviePy VideoClip.
        This function return VRS frame in time alignment with the time {t} request of moviePy.
        - If a frame is not present as a given expected time, we count the frame as missing/dropped and return the last valid frame.
        """

        progress_bar = tqdm(self.iter_data_, desc="Exporting Frames", unit="frame")

        try:
            for obj in progress_bar:
                # We get a new image from the queue
                (
                    self.last_valid_frame_,
                    self.last_timestamp_,
                ) = self.sample_frame_and_timestamp(obj)

                frame_rgb = cv2.cvtColor(self.last_valid_frame_, cv2.COLOR_BGR2RGB)

                # Save the frame to a file
                frame_name = f"frame_{self.frame_count}.jpg"  # Change extension based on your requirement
                cv2.imwrite(f"{self.output_folder_}/{frame_name}", frame_rgb)

                if self.eyegaze_data:
                    eye_gaze_pts = self.log_eye_gaze(self.last_timestamp_)

                    if eye_gaze_pts is not None:
                        x, y = int(eye_gaze_pts[0]), int(eye_gaze_pts[1])

                        # Optionally, you can also save the timestamp along with the frame
                        with open(f"{self.output_folder_}/gaze_coords.txt", "a") as file:
                            file.write(f"{frame_name}: {x},{y}\n")

                self.frame_count += 1
            return self.frame_count

        except StopIteration:
            print("We have exhausted the VRS stream.")
            progress_bar.close()
            return self.frame_count


def main():
    args = parse_args()

    if args.downsample < 1:
        raise ValueError(
            f"Invalid downsample value: {args.downsample}. Must be greater than or equal to 1"
        )
    ##
    # Prepare the Vrs2FramesPyFrameConverter and configure a moviePy video clip
    frame_converter = Vrs2FramesPyFrameConverter(args.vrs, args.eyegaze, args.downsample, args.output_folder)
    # Create a VideoClip of the desired duration, and using the Vrs Source
    frame_count = frame_converter.export_frames()

    print(f"Extracted {frame_count} frames!")

if __name__ == "__main__":
    main()
