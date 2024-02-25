import math
import pybullet

from UR5_Inverse_Kinematics import UR5_Inverse_Kinematics_Simulation
from pybullet_draw_display import set_display_lifetime, draw_coordinate

disp_given_target_orientation = False
given_target_orientation = [[0, 0, 0, 1], [0, 0, 0, 1]]

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
        from scipy.spatial.transform import Rotation
        import numpy
        from math_utils import unwind_angles

        while True:
            if disp_given_target_orientation:
                target_angle = [
                    pybullet.readUserDebugParameter(param_id[1])
                    for param_id in joint_parameters
                ]
                ang = simulation.end_effector_inverse_kinematics_last3dof(given_target_orientation)
                target_angle[3:6] = [angle for angle in ang[0]]
                target_angle[9:12] = [angle for angle in ang[1]]
                simulation.step_simulation(target_angle)
                simulation.draw_end_effector_coordinate(given_target_orientation)
            else:
                simulation.step_simulation(
                    [
                        pybullet.readUserDebugParameter(param_id[1])
                        for param_id in joint_parameters
                    ]
                )
                base = Rotation.from_quat(simulation.get_link_orientation_quaternion(4)).as_matrix()
                transform_base_4_5 = numpy.matrix(
                    numpy.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                )
                ee_ori = numpy.matrix(base) @ transform_base_4_5 @ numpy.matrix(
                    Rotation.from_euler(seq="yzy", degrees=False, angles=[-simulation.get_joint_angle_rad(i) for i in
                                                                          [5, 6, 7]]).as_matrix().transpose())
                draw_coordinate(origin_position=[0, -2, 2],
                                orientation_quaternion=Rotation.from_matrix(ee_ori).as_quat(canonical=True).tolist())
                draw_coordinate(origin_position=[0, 1, 2],
                                orientation_quaternion=simulation.get_link_orientation_quaternion(7))
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
                angle = Rotation.from_matrix(ee_ori.transpose() @ numpy.matrix(base) @ transform_base_4_5).as_euler(
                    seq="yzy", degrees=False) * -1
                angle0 = [-unwind_angles(0, ang) for ang in angle.tolist()]

                angle1 = angle.tolist()
                angle1[0] = angle1[0] - math.pi
                angle1[1] = -angle1[1]
                angle1[2] = angle1[2] - math.pi
                angle1 = [-unwind_angles(0, ang) for ang in angle1]

                angle2 = angle.tolist()
                angle2[0] = math.pi - angle2[0]
                angle2[2] = math.pi - angle2[2]
                angle2 = [-unwind_angles(0, ang) for ang in angle2]

                max_angle = [max(angle0), max(angle1), max(angle2)]
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
