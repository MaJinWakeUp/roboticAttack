#!/usr/bin/env python3
"""Run an SO-100/SO-101 follower arm from the Pi0.5 HTTP server.

The client reads six joint positions plus the required up and wrist cameras,
checks the remote checkpoint schema, and requests action chunks. Robot execution is
disabled unless ``--execute`` is supplied; when enabled, an interactive safety
confirmation is required by default.
"""

from experiments.robot import lerobot_smolvla_so101_client as shared_client


def main() -> None:
    shared_client.LOGGER = shared_client.logging.getLogger("lerobot_pi05_so101_client")
    shared_client.main(
        description=__doc__,
        default_model_name="Pi0.5",
        default_robot_id="pi05_so101_client",
        expected_model_type="pi05",
        default_camera1_key="up",
        default_camera2_key="wrist",
        default_action_chunk_steps=10,
        require_all_server_images=True,
    )


if __name__ == "__main__":
    main()
