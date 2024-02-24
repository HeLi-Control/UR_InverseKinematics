import math
import pybullet

from UR5_Inverse_Kinematics import UR5_Inverse_Kinematics_Simulation
from pybullet_draw_display import draw_coordinate


if __name__ == "__main__":
    simulation = UR5_Inverse_Kinematics_Simulation(
        urdf_file="./ur_description/ur5_robot_hand.urdf", show_gui=True
    )
    joint_parameters = [
        [
            pybullet.addUserDebugParameter(
                paramName=f"Show Joint {joint} coordination",
                rangeMin=1,
                rangeMax=0,
                startValue=1,
            ),
            pybullet.addUserDebugParameter(
                paramName=f"Joint {joint} angle",
                rangeMin=-2 * math.pi,
                rangeMax=2 * math.pi,
                startValue=0,
            ),
            joint,
        ]
        for joint in simulation.available_joints_indices
    ]
    button_initial_value = [1.0 for _ in simulation.available_joints_indices]
    show_coordinate_flag = [False for _ in simulation.available_joints_indices]
    try:
        while True:
            simulation.step_simulation(
                [
                    pybullet.readUserDebugParameter(param_id[1])
                    for param_id in joint_parameters
                ]
            )
            for index in range(simulation.available_joints_num):
                if (
                    pybullet.readUserDebugParameter(joint_parameters[index][0])
                    != button_initial_value[index]
                ):
                    button_initial_value[index] = button_initial_value[index] + 1
                    show_coordinate_flag[index] = not show_coordinate_flag[index]
                if show_coordinate_flag[index]:
                    draw_coordinate(
                        origin_position=simulation.get_link_position_xyz(
                            joint_parameters[index][2]
                        ),
                        orientation_quaternion=simulation.get_link_orientation_quaternion(
                            joint_parameters[index][2]
                        ),
                    )

    except KeyboardInterrupt:
        pybullet.disconnect(simulation.client)
