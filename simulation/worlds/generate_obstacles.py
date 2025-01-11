from typing import Tuple
import numpy as np
import random
import pathlib



SDF_WORLD_PATH = "/root/sim_ws/src/psw_challenge/worlds/"
SDF_WORLD_NAME = "static_landing.world"
OBSTACLES = [ "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Pine Tree", 
              "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Oak tree", 
              "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Telephone pole", 
              "Wooden Peg"]
MY_OBSTACLES = OBSTACLES[:3]  # Selecting obstacles
NO_OBSTACLES = 17  # export GZ_SIM_RESOURCE_PATH=~/PX4-Autopilot/Tools/simulation/gz/models:~/PX4-Autopilot/Tools/simulation/gz/worlds
OBSTACLES_AREA = ((30, 80), (-25, 35))
MIN_OBSTACLES_DISTANCE = 10


def obst_sdf_code(obstacle: str, pose: Tuple[float], rotation: float, name: str) -> str:
    code = f"    <include><uri>{obstacle}</uri><name>{name}</name><pose>{pose[0]} {pose[1]} 0 0 0 {rotation}</pose></include>\n"
    return code

def main():
    with open(SDF_WORLD_PATH + SDF_WORLD_NAME, "r") as f:
        code_lines = f.readlines()

    info = "    <!-- Randomly generated obstacles -->\n"
    info2 =f"    <!-- Number of obstacles = {NO_OBSTACLES} -->\n"
    if info in code_lines:
        del code_lines[-4-int(code_lines[code_lines.index(info) + 1][-7:-5]):-2]

    code_lines.insert(-2, info)
    code_lines.insert(-2, info2)

    positions = list()
    for i in range(NO_OBSTACLES):
        pose = (random.uniform(OBSTACLES_AREA[0][0], OBSTACLES_AREA[0][1]), random.uniform(OBSTACLES_AREA[1][0], OBSTACLES_AREA[1][1]))

        dists = [np.linalg.norm(np.subtract(prev_pose, pose)) for prev_pose in positions]
        while any([dist < MIN_OBSTACLES_DISTANCE for dist in dists]):
            pose = (random.uniform(OBSTACLES_AREA[0][0], OBSTACLES_AREA[0][1]), random.uniform(OBSTACLES_AREA[1][0], OBSTACLES_AREA[1][1]))
            dists = [np.linalg.norm(np.subtract(prev_pose, pose)) for prev_pose in positions]

        rotation = random.uniform(-np.pi, np.pi)
        obstacle = random.choice(MY_OBSTACLES)

        code = obst_sdf_code(obstacle, pose, rotation, f"obstacle_{i+1}")
        positions.append(pose)

        code_lines.insert(-2, code)

    with open(f"{SDF_WORLD_PATH}{SDF_WORLD_NAME}", "w") as f:
        f.writelines(code_lines)


if __name__ == "__main__":
    main()