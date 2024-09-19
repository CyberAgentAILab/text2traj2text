ZERO_SHOT_PROMPT = """{bos_token}You will be given a person's trajectory in the supermarket. As a good AI, please generate a Caption that describes the trajectory.

Trajectory: {trajectory}"""


FEW_SHOT_PROMPT = """{bos_token}You will be given several pairs of trajectories and contextual captions as examples. As a good AI, please generate a caption that describes the trajectory based on the examples.

{example}

Trajectory: {trajectory}"""
