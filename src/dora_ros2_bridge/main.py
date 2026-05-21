# Copyright 2026 Enactic, Inc.
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

"""Dora-to-ROS2 Bridge Node.

This node acts as a state holder and message translator between the Dora dataflow
and ROS 2 control interfaces. It subscribes to joint position and camera inputs
from the Dora graph and forwards them to the appropriate ROS 2 controllers.

Current use cases:
  - Teleoperation: relay joint commands from a leader to the robot controllers
  - Data collection: forward states for rosbag recording

This bridge will be updated as teleoperation and data collection requirements evolve.

Inputs (Dora):
  - position_left  : float32[8] - left arm joints (7) + gripper (1)
  - position_right : float32[8] - right arm joints (7) + gripper (1)
  - image          : uint8[]    - JPEG-encoded image

Outputs (ROS 2):
  - /left_joint_trajectory_controller/joint_trajectory
  - /right_joint_trajectory_controller/joint_trajectory
  - /left_gripper_controller/joint_trajectory
  - /right_gripper_controller/joint_trajectory
  - /camera/image/compressed
"""

import dora
import pyarrow as pa


def main():
    """Run the Dora-to-ROS 2 bridge."""
    # --- 1. ROS 2 Setup ---
    context = dora.Ros2Context()
    options = dora.Ros2NodeOptions(rosout=True)
    node = context.new_node("dora_to_ros2", "/openarm", options)

    qos_arm = dora.Ros2QosPolicies(reliable=True)

    qos_cam = dora.Ros2QosPolicies(reliable=False)

    # --- 2. Define Publishers ---
    p_l_arm = node.create_publisher(
        node.create_topic(
            "/left_joint_trajectory_controller/joint_trajectory",
            "trajectory_msgs/JointTrajectory",
            qos_arm,
        )
    )
    p_r_arm = node.create_publisher(
        node.create_topic(
            "/right_joint_trajectory_controller/joint_trajectory",
            "trajectory_msgs/JointTrajectory",
            qos_arm,
        )
    )
    p_l_grp = node.create_publisher(
        node.create_topic(
            "/left_gripper_controller/joint_trajectory",
            "trajectory_msgs/JointTrajectory",
            qos_arm,
        )
    )
    p_r_grp = node.create_publisher(
        node.create_topic(
            "/right_gripper_controller/joint_trajectory",
            "trajectory_msgs/JointTrajectory",
            qos_arm,
        )
    )

    p_cam = node.create_publisher(
        node.create_topic(
            "/camera/image/compressed", "sensor_msgs/CompressedImage", qos_cam
        )
    )

    # --- 3. Pre-defined Constants & Templates (Object Reuse for Speed) ---
    EMPTY_F64 = pa.scalar([], pa.list_(pa.float64()))
    SEC_0 = pa.scalar(0, pa.int32())
    NSEC_0 = pa.scalar(0, pa.uint32())
    NSEC_WAIT = pa.scalar(0, pa.uint32())

    NAMES_L_ARM = [f"openarm_left_joint{i + 1}" for i in range(7)]
    NAMES_R_ARM = [f"openarm_right_joint{i + 1}" for i in range(7)]
    NAMES_L_GRP = ["openarm_left_finger_joint1"]
    NAMES_R_GRP = ["openarm_right_finger_joint1"]

    joint_msg = {
        "header": {"stamp": {"sec": SEC_0, "nanosec": NSEC_0}, "frame_id": ""},
        "joint_names": None,
        "points": [
            {
                "positions": None,
                "velocities": EMPTY_F64,
                "accelerations": EMPTY_F64,
                "effort": EMPTY_F64,
                "time_from_start": {"sec": SEC_0, "nanosec": NSEC_WAIT},
            }
        ],
    }

    compressed_img_msg = {
        "header": {"stamp": {"sec": SEC_0, "nanosec": NSEC_0}, "frame_id": "world"},
        "format": "jpeg",
        "data": None,
    }

    # --- 4. Dora Loop ---
    dora_node = dora.Node()
    for event in dora_node:
        if event["type"] != "INPUT":
            continue

        eid = event["id"]
        value = event["value"]

        # --- Case A: Camera Image (JPEG Pass-through) ---
        if eid == "image":
            img_data = pa.scalar(value.cast(pa.uint8()), pa.list_(pa.uint8()))
            compressed_img_msg["data"] = img_data
            p_cam.publish(pa.array([compressed_img_msg]))
            continue

        # --- Case B: Joint Positions ---
        if eid == "position_left":
            pub_arm, pub_grp, name_arm, name_grp = (
                p_l_arm,
                p_l_grp,
                NAMES_L_ARM,
                NAMES_L_GRP,
            )
        elif eid == "position_right":
            pub_arm, pub_grp, name_arm, name_grp = (
                p_r_arm,
                p_r_grp,
                NAMES_R_ARM,
                NAMES_R_GRP,
            )
        else:
            continue

        value_double = value.cast(pa.float64())
        joint_msg["joint_names"] = name_arm
        joint_msg["points"][0]["positions"] = pa.scalar(
            value_double[:7], pa.list_(pa.float64())
        )
        pub_arm.publish(pa.array([joint_msg]))

        if len(value_double) >= 8:
            joint_msg["joint_names"] = name_grp
            joint_msg["points"][0]["positions"] = pa.scalar(
                value_double[7:8], pa.list_(pa.float64())
            )
            pub_grp.publish(pa.array([joint_msg]))


if __name__ == "__main__":
    main()
