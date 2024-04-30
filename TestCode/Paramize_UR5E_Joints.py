import math
import pybullet

from InverseKinematics.UR5E_Inverse_Kinematics import ur5e_robot_inverse_kinematics
from Utils.pybullet_draw_display import set_display_lifetime, draw_coordinate

import matplotlib.pyplot as plt

disp_given_target_orientation = True
given_target_orientation = [0.0, 0.0, 0.0, 1.0]
show_set_angle_plot = False

if __name__ == "__main__":
    simulation = ur5e_robot_inverse_kinematics(
        urdf_file="../RobotDescription/ur5e/ur5e.urdf", show_gui=True
    )
    start_angle = [-1.058220386505127, -0.7936654090881348, 1.3889145851135254, -2.546343557215553, 1.0582203778365116,
                   -3.085593291984878e-08]
    # Create interacting debug parameters
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
    set_display_lifetime(0.01)
    set_ang = [[simulation.get_joint_angle_rad(i) for i in (4, 5, 6)]]
    draw_plot_data = [[], [], []]
    while_index = 1
    try:
        while True:
            if disp_given_target_orientation:
                target_angle = [
                    pybullet.readUserDebugParameter(param_id[1]) for param_id in joint_parameters
                ]
                now_angle = [[simulation.get_joint_angle_rad(i) for i in (4, 5, 6)]]
                target_angle[3:] = simulation.end_effector_inverse_kinematics_last3dof(
                    given_target_orientation, now_angle=set_ang, random_select=True
                )[0]
                set_ang = [target_angle[3:]]
                # logger.debug('target angle = ' + str(target_angle))
                simulation.step_simulation(target_angle)
                simulation.draw_end_effector_coordinate(given_target_orientation)
                # Draw set angle plot
                if show_set_angle_plot:
                    plt.clf()
                    for i in range(3):
                        draw_plot_data[i].append(set_ang[0][i])
                        if len(draw_plot_data[i]) != while_index:
                            draw_plot_data[i].pop(0)
                    plt.plot([i + 1 for i in range(while_index)], draw_plot_data[0], label='joint4')
                    plt.plot([i + 1 for i in range(while_index)], draw_plot_data[1], label='joint5')
                    plt.plot([i + 1 for i in range(while_index)], draw_plot_data[2], label='joint6')
                    plt.legend()
                    plt.ylabel("Set angle(rad)")
                    plt.pause(0.001)
                    plt.ioff()
                    while_index = while_index + 1 if while_index < 50 else while_index
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
                        orientation_quaternion=simulation.get_link_orientation_quaternion(joint_parameters[index][2]),
                    )
    except KeyboardInterrupt:
        pybullet.disconnect(simulation.client)
