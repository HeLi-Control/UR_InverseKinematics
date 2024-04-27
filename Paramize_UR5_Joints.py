import math
import pybullet

from UR5_Inverse_Kinematics import UR5_Inverse_Kinematics_Simulation
from pybullet_draw_display import set_display_lifetime, draw_coordinate

disp_given_target_orientation = True
given_target_orientation = [[0, 0, 0, 1], [0, 0, 0, 1]]

if __name__ == "__main__":
    simulation = UR5_Inverse_Kinematics_Simulation(
        urdf_file="./RobotDescription/ur_description/ur5_robot_hand.urdf", show_gui=True
    )
    start_angle = [
        0.0, 0.0, 0.0, -2.186088266204923, 1.0469676627563713, -0.6158551755776115,
        0.0, 0.0, 0.0, 2.186838581440922, 2.0937051407187877, -0.6143539750849345,
    ]
    joint_parameters = [
        [
            pybullet.addUserDebugParameter(
                paramName=f"Show Joint {joint} coordination", rangeMin=1, rangeMax=0, startValue=1,
            ),
            pybullet.addUserDebugParameter(
                paramName=f"Joint {joint} angle", rangeMin=-2 * math.pi, rangeMax=2 * math.pi,
                startValue=start_angle[index],
            ),
            joint,
        ]
        for joint, index in zip(
            simulation.available_joints_indices, [i for i in range(len(simulation.available_joints_indices))],
        )
    ]
    button_initial_value = [1.0 for _ in simulation.available_joints_indices]
    show_coordinate_flag = [False for _ in simulation.available_joints_indices]
    set_ang = [[0, 0, 0], [0, 0, 0]]
    set_display_lifetime(0.01)
    try:
        while True:
            if disp_given_target_orientation:
                target_angle = [pybullet.readUserDebugParameter(param_id[1]) for param_id in joint_parameters]
                now_ang = [simulation.get_joint_angle_rad(index) for index in simulation.available_joints_indices]
                ang = simulation.end_effector_inverse_kinematics_last3dof(
                    given_target_orientation, now_angle=set_ang, random_select=True
                )
                target_angle[3:6] = ang[0]
                target_angle[9:12] = ang[1]
                set_ang = [target_angle[3:6], target_angle[9:12]]
                simulation.step_simulation(target_angle)
                simulation.draw_end_effector_coordinate(given_target_orientation)
            else:
                simulation.step_simulation(
                    [pybullet.readUserDebugParameter(param_id[1]) for param_id in joint_parameters]
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
                        origin_position=simulation.get_link_position_xyz(joint_parameters[index][2]),
                        orientation_quaternion=simulation.get_link_orientation_quaternion(
                            joint_parameters[index][2]
                        ),
                    )

    except KeyboardInterrupt:
        pybullet.disconnect(simulation.client)
