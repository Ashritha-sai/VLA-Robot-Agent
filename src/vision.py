"""
VisionModule - Camera-based scene perception using PyBullet rendering.

Provides image capture and basic object detection as an alternative
(or supplement) to the structured ``get_scene_description()`` method.
"""

import pybullet as p
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VisionModule:
    """Camera-based perception for the tabletop environment."""

    # Default virtual camera parameters
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_FOV = 60.0
    DEFAULT_NEAR = 0.1
    DEFAULT_FAR = 3.0

    def __init__(self, env, width: int = None, height: int = None):
        """
        Args:
            env: A TableTopEnv instance.
            width: Image width in pixels (default 640).
            height: Image height in pixels (default 480).
        """
        self.env = env
        self.width = width or self.DEFAULT_WIDTH
        self.height = height or self.DEFAULT_HEIGHT

        # Camera extrinsics: look at table centre from above and to the side
        self.camera_target = [0.5, 0.0, 0.05]
        self.camera_distance = 1.0
        self.camera_yaw = 45
        self.camera_pitch = -30

    def capture_image(self) -> Dict[str, np.ndarray]:
        """
        Capture an RGB-D image from the virtual camera.

        Returns:
            Dict with keys:
                ``rgb``: (H, W, 3) uint8 array
                ``depth``: (H, W) float32 array (metres)
                ``segmentation``: (H, W) int32 array (body IDs)
        """
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.DEFAULT_FOV,
            aspect=self.width / self.height,
            nearVal=self.DEFAULT_NEAR,
            farVal=self.DEFAULT_FAR,
        )

        _, _, rgb, depth, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        rgb_array = np.array(rgb, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
        depth_array = np.array(depth, dtype=np.float32).reshape(self.height, self.width)
        seg_array = np.array(seg, dtype=np.int32).reshape(self.height, self.width)

        return {
            "rgb": rgb_array,
            "depth": depth_array,
            "segmentation": seg_array,
        }

    def detect_objects(self) -> List[Dict]:
        """
        Detect objects in the scene using the segmentation mask.

        Returns:
            List of dicts with keys ``name``, ``body_id``, ``pixel_count``,
            ``centroid_uv`` (approximate image centroid).
        """
        images = self.capture_image()
        seg = images["segmentation"]

        # Find unique body IDs in the segmentation mask
        unique_ids = set(np.unique(seg).tolist())
        # Remove background (-1) and ground plane / table
        unique_ids.discard(-1)
        unique_ids.discard(self.env.plane_id)
        unique_ids.discard(self.env.table_id)
        unique_ids.discard(self.env.robot_id)

        # Map body_id -> object name
        id_to_name = {v: k for k, v in self.env.objects.items()}

        results = []
        for body_id in unique_ids:
            mask = seg == body_id
            pixel_count = int(np.sum(mask))
            if pixel_count == 0:
                continue
            ys, xs = np.where(mask)
            centroid_u = float(np.mean(xs))
            centroid_v = float(np.mean(ys))
            name = id_to_name.get(body_id, f"unknown_{body_id}")
            results.append({
                "name": name,
                "body_id": body_id,
                "pixel_count": pixel_count,
                "centroid_uv": [centroid_u, centroid_v],
            })

        results.sort(key=lambda d: d["pixel_count"], reverse=True)
        return results

    def get_scene_description_from_vision(self) -> Dict:
        """
        Build a scene description using camera perception.

        Combines standard structured data (robot state, object positions)
        with vision-derived detection results. This can be used as a
        drop-in replacement for ``env.get_scene_description()``.

        Returns:
            Scene description dict compatible with VLAAgent._build_prompt().
        """
        # Start with the standard structured description
        scene = self.env.get_scene_description()

        # Augment with vision detections
        detections = self.detect_objects()
        detected_names = {d["name"] for d in detections}

        # Add pixel count info to objects that were visually detected
        for obj in scene["objects"]:
            detection = next((d for d in detections if d["name"] == obj["name"]), None)
            if detection:
                obj["visible"] = True
                obj["pixel_count"] = detection["pixel_count"]
            else:
                obj["visible"] = False
                obj["pixel_count"] = 0

        return scene
