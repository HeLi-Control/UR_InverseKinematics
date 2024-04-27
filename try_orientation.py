import math
import pybullet

from UR5_Inverse_Kinematics import UR5_Inverse_Kinematics_Simulation
from pybullet_draw_display import set_display_lifetime, draw_coordinate

from scipy.spatial.transform import Rotation
import numpy
from math_utils import unwind_angles

from loguru import logger


check_left = True


def get_yzy_euler_angles_from_rotation_matrix(target_rotation_matrix: numpy.matrix) -> list[float]:
    ret_angles = Rotation.from_matrix(target_rotation_matrix).as_euler(seq="yzy", degrees=False) * -1
    # euler_angle = [[unwind_angles(0, ang) for ang in ret_angles.tolist()]] * 3
    euler_angle = [[unwind_angles(0, ang) for ang in ret_angles.tolist()] for _ in range(3)]

    euler_angle[1][0] = euler_angle[1][0] - math.pi
    euler_angle[1][1] = -euler_angle[1][1]
    euler_angle[1][2] = euler_angle[1][2] - math.pi
    euler_angle[1] = [unwind_angles(0, ang) for ang in euler_angle[1]]

    euler_angle[2][0] = math.pi - euler_angle[2][0]
    euler_angle[2][2] = math.pi - euler_angle[2][2]
    euler_angle[2] = [unwind_angles(0, ang) for ang in euler_angle[2]]

    max_angle = [max(numpy.abs(numpy.array(angle)).tolist()) for angle in euler_angle]
    logger.info(euler_angle[0])
    return euler_angle[max_angle.index(min(max_angle))]


if __name__ == "__main__":
    simulation = UR5_Inverse_Kinematics_Simulation(
        urdf_file="./ur_description/ur5_robot_hand.urdf", show_gui=True
    )
    start_angle = [0.0, 0.0, 0.0, -2.186088266204923, 1.0469676627563713, -0.6158551755776115,
                   0.0, 0.0, 0.0, 2.186838581440922, 2.0937051407187877, -0.6143539750849345]
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
                startValue=start_angle[index],
            ),
            joint,
        ]
        for joint, index in
        zip(simulation.available_joints_indices, [i for i in range(len(simulation.available_joints_indices))])
    ]
    button_initial_value = [1.0 for _ in simulation.available_joints_indices]
    show_coordinate_flag = [False for _ in simulation.available_joints_indices]
    set_display_lifetime(0.01)
    try:
        while True:
            if check_left:
                # Left Arm
                logger.debug('<' * 50 + 'Left' + '>' * 50)
                # Wrist base coordinate
                base = Rotation.from_quat(simulation.get_link_orientation_quaternion(4)).as_matrix()
                transform_base_4_5 = numpy.matrix(
                    numpy.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                )
                # Calculate ee orientation from euler angle
                ee_ori = numpy.matrix(base) @ transform_base_4_5 @ numpy.matrix(
                    Rotation.from_euler(seq="yzy", degrees=False, angles=[-simulation.get_joint_angle_rad(i) for i in
                                                                          [5, 6, 7]]).as_matrix().transpose())
                draw_coordinate(origin_position=[0, -2, 2],
                                orientation_quaternion=Rotation.from_matrix(ee_ori).as_quat(canonical=True).tolist())
                # Get ee orientation directly
                draw_coordinate(origin_position=[0, 1, 2],
                                orientation_quaternion=simulation.get_link_orientation_quaternion(7))
                # Calculate ee orientation from rotation matrix
                angle5 = simulation.get_joint_angle_rad(5)
                angle6 = simulation.get_joint_angle_rad(6)
                angle7 = simulation.get_joint_angle_rad(7)
                ee_ori = numpy.matrix(base) @ transform_base_4_5 @ numpy.matrix(numpy.array(
                    [[math.cos(angle5), 0, math.sin(angle5)], [0, 1, 0], [-math.sin(angle5), 0, math.cos(angle5)]]
                )) @ numpy.matrix(numpy.array(
                    [[math.cos(angle6), -math.sin(angle6), 0], [math.sin(angle6), math.cos(angle6), 0], [0, 0, 1]]
                )) @ numpy.matrix(numpy.array(
                    [[math.cos(angle7), 0, math.sin(angle7)], [0, 1, 0], [-math.sin(angle7), 0, math.cos(angle7)]]
                ))
                draw_coordinate(origin_position=[0, -1, 2],
                                orientation_quaternion=Rotation.from_matrix(ee_ori).as_quat(canonical=True).tolist())
                # Calculate joint angle from given rotation
                rotation_matrix = ee_ori.transpose() @ numpy.matrix(base) @ transform_base_4_5
                final_euler_angle = get_yzy_euler_angles_from_rotation_matrix(rotation_matrix)
                logger.debug(final_euler_angle)
            else:
                # Right Arm
                logger.debug('<' * 50 + 'Right' + '>' * 50)
                # Wrist base coordinate
                base = Rotation.from_quat(simulation.get_link_orientation_quaternion(13)).as_matrix()
                transform_base_13_14 = numpy.matrix(
                    numpy.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                )
                # Calculate ee orientation from euler angle
                ee_ori = numpy.matrix(base) @ transform_base_13_14 @ numpy.matrix(
                    Rotation.from_euler(seq="yzy", degrees=False, angles=[-simulation.get_joint_angle_rad(i) for i in
                                                                          [14, 15, 16]]).as_matrix().transpose())
                draw_coordinate(origin_position=[0, -2, 2],
                                orientation_quaternion=Rotation.from_matrix(ee_ori).as_quat(canonical=True).tolist())
                # Get ee orientation directly
                draw_coordinate(origin_position=[0, 1, 2],
                                orientation_quaternion=simulation.get_link_orientation_quaternion(16))
                # Calculate ee orientation from rotation matrix
                angle14 = simulation.get_joint_angle_rad(14)
                angle15 = simulation.get_joint_angle_rad(15)
                angle16 = simulation.get_joint_angle_rad(16)
                ee_ori = numpy.matrix(base) @ transform_base_13_14 @ numpy.matrix(numpy.array(
                    [[math.cos(angle14), 0, math.sin(angle14)], [0, 1, 0], [-math.sin(angle14), 0, math.cos(angle14)]]
                )) @ numpy.matrix(numpy.array(
                    [[math.cos(angle15), -math.sin(angle15), 0], [math.sin(angle15), math.cos(angle15), 0], [0, 0, 1]]
                )) @ numpy.matrix(numpy.array(
                    [[math.cos(angle16), 0, math.sin(angle16)], [0, 1, 0], [-math.sin(angle16), 0, math.cos(angle16)]]
                ))
                draw_coordinate(origin_position=[0, -1, 2],
                                orientation_quaternion=Rotation.from_matrix(ee_ori).as_quat(canonical=True).tolist())
                # Calculate joint angle from given rotation
                rotation_matrix = ee_ori.transpose() @ numpy.matrix(base) @ transform_base_13_14
                final_euler_angle = get_yzy_euler_angles_from_rotation_matrix(rotation_matrix)
                logger.info(final_euler_angle)

            # Joint angle control
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
            simulation.step_simulation(
                [
                    pybullet.readUserDebugParameter(param_id[1])
                    for param_id in joint_parameters
                ]
            )

    except KeyboardInterrupt:
        pybullet.disconnect(simulation.client)
