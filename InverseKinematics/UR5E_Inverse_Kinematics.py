import math

import numpy as np
import h5py
from scipy.spatial.transform import Rotation
import pybullet

from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from InverseKinematics.UR5_Inverse_Kinematics import ur5_robot_inverse_kinematics, read_frame_demonstrate_data

from Utils.math_utils import get_yzy_euler_angles_from_rotation_matrix, unwind_angle_list, cvt_target
from Utils.pybullet_draw_display import set_display_lifetime, disp_human_demonstrate_arm, draw_coordinate

global_z_offset = 1.0
given_fixed_orientation = False
fixed_orientation = [1.0, 0.0, 0.0, 0.0]
draw_end_effector_coordinate = True
calculate_wrist_orientation_self = True
plot_angles = True
calculate_reference_speed = True

if calculate_reference_speed:
    import statsmodels.api as sm


class ur5e_robot_inverse_kinematics(ur5_robot_inverse_kinematics):
    def __init__(self, urdf_file: str, ik_use_world_orientation=True, show_gui=False, default_scale=None):
        self.default_scale = [1.2, 1.7] if default_scale is None else default_scale
        super(ur5e_robot_inverse_kinematics, self).__init__(urdf_file, ik_use_world_orientation, show_gui,
                                                            self.default_scale)
        self.end_effector_joint_index = [6]

    # .available_joints_indices = [1, 2, 3, 4, 5, 6]
    # .available_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
    #    'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    @property
    def arm_base_position(self) -> list[float]:
        return self.get_link_position_xyz(self.available_joints_indices[0])

    @property
    def ee_orientation_quaternion(self) -> list[float]:
        return self.get_link_orientation_quaternion(self.end_effector_joint_index[0])

    def end_effector_inverse_kinematics_last3dof(
            self,
            target_orientations: list[float],
            now_angle: list[list[float]],
            random_select=False
    ) -> list[list[float]]:
        def quaternion_2_numpy_matrix(quaternion: list[float]) -> np.matrix:
            return np.matrix(np.array(Rotation.from_quat(quaternion).as_matrix()))

        def unwind_euler_angle_lists(center_angle: list[float], target_angles: list[list[float]]) -> list[list[float]]:
            return [
                unwind_angle_list([0] * len(center_angle), unwind_angle_list(center_angle, target_angle),
                                  period=math.pi * 4, step_size=math.pi * 2)
                for target_angle in target_angles
            ]

        if self.ik_use_world_orientation:
            def get_wrist_last3dof(
                    ee_ori: np.matrix,
                    base: np.matrix,
                    base_transform=np.matrix(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])),
            ) -> list[list[float]]:
                rotation_matrix = ee_ori.transpose() @ base @ base_transform
                return get_yzy_euler_angles_from_rotation_matrix(rotation_matrix)

            # Wrist inverse kinematics
            wrist_base_ori = [quaternion_2_numpy_matrix(self.get_link_orientation_quaternion(3))]
            wrist_target_ori = quaternion_2_numpy_matrix(target_orientations)
            last3dof_ang = [unwind_euler_angle_lists(
                now_angle[0], get_wrist_last3dof(wrist_target_ori, wrist_base_ori[0]))]
        else:
            last3dof_ang = [unwind_euler_angle_lists(
                now_angle[0], get_yzy_euler_angles_from_rotation_matrix(
                    quaternion_2_numpy_matrix(target_orientations) @ np.matrix(
                        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))))]
        last3dof_ang = last3dof_ang[0:1]
        # Select the proper result
        if random_select:
            return last3dof_ang[0]
        angle_error = np.sum(
            np.abs(np.array(last3dof_ang) - np.array([[ang] for ang in now_angle])), axis=2,
        )
        min_index = np.argmin(angle_error, axis=1).tolist()
        return [wrist_angles[index] for wrist_angles, index in zip(last3dof_ang, min_index)]
        # return [wrist_angles[0] for wrist_angles in last3dof_ang]

    @staticmethod
    def arm_calculate_inverse_kinematics(target_position: list[list[float]]) -> list[float]:
        # WARNING: When the projection locates around the origin or the YZ plane, the result comes ill-posed.
        def safe_arccos(x: float) -> float:
            return 0.0 if x > 0.999 else (math.pi if x < -0.999 else math.acos(x))

        wrist_pos = (np.array(target_position[0]) - np.array(target_position[2])).tolist()
        elbow_pos = (np.array(target_position[1]) - np.array(target_position[2])).tolist()
        # Get first yaw angle
        yaw_ang = math.atan2(wrist_pos[1], wrist_pos[0]) + math.pi
        # Get next two pitch angles
        r_target = np.linalg.norm(np.array(wrist_pos))
        link_length = [np.linalg.norm(np.array(elbow_pos)),
                       np.linalg.norm(np.array(wrist_pos) - np.array(elbow_pos))]
        cos_pitch1 = (r_target ** 2 + link_length[0] ** 2 - link_length[1] ** 2) / (2 * r_target * link_length[0])
        cos_pitch2 = (link_length[0] ** 2 + link_length[1] ** 2 - r_target ** 2) / (2 * link_length[0] * link_length[0])
        r_xy = (wrist_pos[0] ** 2 + wrist_pos[1] ** 2) ** 0.5
        pitch1_ang = -(math.pi - (math.atan2(wrist_pos[2], r_xy) + safe_arccos(cos_pitch1)))
        pitch2_ang = -safe_arccos(-cos_pitch2)
        return [yaw_ang, pitch1_ang, pitch2_ang, 0.0, 0.0, 0.0]

    def calculate_inverse_kinematics(
            self,
            target_positions: list[list[float]],
            target_orientations: list[float],
            use_elbow_pos=True,
            use_self_kinematics=True
    ) -> list[float]:
        arm_now_ang = [self.get_joint_angle_rad(index) for index in self.available_joints_indices]
        # Arm inverse kinematics
        _angles = self.arm_calculate_inverse_kinematics(
            target_position=[target_positions[target_joint] for target_joint in [2, 1, 0]]) \
            if use_self_kinematics else self.calculate_inverse_kinematics_without_orientation(
            target_joints_indices=[6, 3] if use_elbow_pos else [6],
            target_positions=[target_positions[target_joint] for target_joint in ([2, 1] if use_elbow_pos else [2])])
        # Wrist orientation inverse kinematics
        ang = self.end_effector_inverse_kinematics_last3dof(
            target_orientations, now_angle=[self.ctrl_command[3:]] if self.ctrl_command is not None else [[0] * 3],
            random_select=False)
        _angles[3:6] = ang[0]

        return unwind_angle_list(
            [0] * len(_angles), unwind_angle_list(arm_now_ang, _angles), period=4 * math.pi, step_size=2 * math.pi,
        )

    def draw_end_effector_coordinate(self, target_orientation: list[float]) -> None:
        self.draw_single_joint_coordinate(self.end_effector_joint_index[0], target_orientation)

    def get_real_target(
            self, arm_base_position: list[float], _target: list[list[float]], man_scale: list[float]
    ) -> list[list[float]]:
        if not man_scale:
            man_scale = self.default_scale
        target_points = cvt_target(_target, arm_base_position, _man_scale=man_scale)
        disp_human_demonstrate_arm(target_points, draw_bias=[0, -0.6, global_z_offset])
        pybullet.addUserDebugPoints(
            pointPositions=target_points[:3], pointColorsRGB=[[1, 0, 0]] * len(target_points[:3]),
            pointSize=2, lifeTime=0.1,
        )
        return target_points

    def given_demonstrate_data_step_simulation(self, _target_pos: list[list[float]], _target_ori: list[float]) -> bool:
        # Read interact input
        if (self.show_gui and
                (pybullet.readUserDebugParameter(self.display_button_id) != self.display_button_id_last_value)):
            self.display_button_id_last_value = self.display_button_id_last_value + 1
            self.display_demonstrate_flag = not self.display_demonstrate_flag
        interact_scale = [
            pybullet.readUserDebugParameter(self.demonstrate_scale_id[0]),
            pybullet.readUserDebugParameter(self.demonstrate_scale_id[1])
        ] if self.show_gui else []
        # Scale the input demonstrate
        _target_pos = self.get_real_target(simulation.arm_base_position, _target_pos, man_scale=interact_scale)
        # Inverse Kinematics
        angles = simulation.calculate_inverse_kinematics(_target_pos, _target_ori, use_elbow_pos=True)
        # Step simulation
        simulation.step_simulation(angles)
        return self.display_demonstrate_flag if self.show_gui else True


def estimate_speed_given_positions(position: list[float], time_stamp: list[float]) -> float:
    if len(position) != len(time_stamp):
        raise ValueError(f'The data length of variable \'position\' is {len(position)}'
                         f'while which of \'time_stamp\' is {len(time_stamp)}')
    if (len(position) < 20) or not calculate_reference_speed:
        return 0
    else:
        position_new = position[-18:]
        time_stamp_new = time_stamp[-18:]
        weight = np.array([(w + 1) ** 1.2 for w in range(len(position_new))])
        weight = (weight / weight.sum()).tolist()
        fitted = sm.WLS(sm.add_constant(position_new), time_stamp_new, weights=weight).fit()
        return fitted.params.tolist()[0][1]


if __name__ == "__main__":
    disp_human_demonstrate_file = '../DemonstrateData/demo3.pkl'
    # disp_human_demonstrate_file = '../DemonstrateData/humanDemonstrate.h5'
    disp_human_demonstrate_file_ish5 = disp_human_demonstrate_file.endswith('h5')
    # Load demonstrate data
    demonstrate_data = h5py.File(name=disp_human_demonstrate_file, mode="r") if disp_human_demonstrate_file_ish5 \
        else np.load(file=disp_human_demonstrate_file, allow_pickle=True)
    # Start simulation
    simulation = ur5e_robot_inverse_kinematics(urdf_file="../RobotDescription/ur5e/ur5e.urdf",
                                               ik_use_world_orientation=True, show_gui=False, default_scale=[1.6, 1.7])
    set_display_lifetime(0.01)
    # Simulation in loop
    ctrl_angles = []
    ctrl_time = []
    ctrl_angular_speed = []
    start_time = time.time()
    target_angular_speed = [0.0] * 6
    try:
        data_length = len(demonstrate_data) if isinstance(demonstrate_data, list) else len(
            list(demonstrate_data["l_arm"]))
        with (tqdm(total=data_length, unit='Frames') as pbar):
            loop_index = 0
            while True:
                # Read demonstrate data
                target_pos, target_ori = read_frame_demonstrate_data(
                    demonstrate_data, loop_index, calculate_wrist_orientation_self, disp_human_demonstrate_file_ish5,
                    given_fixed_orientation, fixed_orientation)
                target_pos = target_pos[:3]
                target_ori = [target_ori[0]]
                # WARNING: Negative Position Z in both position and orientation, flipped along the YZ plane meanwhile.
                # for i in range(3):
                #     target_pos[i][0] = -target_pos[i][0]
                #     target_pos[i][1] = -target_pos[i][1]
                target_pos[2][0] += 0.1
                target_pos[1][2] += 0.2
                # target_pos[2][2] += 0.3
                target_ori = [list(Rotation.from_matrix(
                    np.matrix(np.diag([-1, -1, 1])) @ Rotation.from_quat(target_ori[0]).as_matrix() @
                    np.matrix(np.diag([-1, 1, -1]))).as_quat(canonical=True))]
                if draw_end_effector_coordinate:
                    # Draw base coordinate
                    draw_coordinate([0, 0, 0], [0, 0, 0, 1])
                    # Draw orientation coordinate
                    simulation.draw_end_effector_coordinate(target_ori[0])
                # Step simulation
                if simulation.given_demonstrate_data_step_simulation(target_pos, target_ori[0]):
                    # Record control command data
                    delta_time = time.time() - start_time
                    ctrl_angles.append(simulation.ctrl_command)
                    ctrl_time.append(delta_time)
                    # TODO: This is the real control command.
                    target_angular_speed = [
                        estimate_speed_given_positions(np.array(ctrl_angles).T.tolist()[joint][-20:],
                                                       ctrl_time[-20:]) for joint
                        in range(simulation.available_joints_num)]
                    now_ang = [simulation.get_joint_angle_rad(joint_index) for joint_index in
                               simulation.available_joints_indices]
                    angular_speed_cmd = (np.array(target_angular_speed) + 0.5 * (
                            np.array(simulation.ctrl_command) - np.array(now_ang))).tolist()
                    ctrl_angular_speed.append(angular_speed_cmd)
                    if len(ctrl_angular_speed) > 1500:
                        ctrl_angles = ctrl_angles[-1500:]
                        ctrl_time = ctrl_time[-1500:]
                        ctrl_angular_speed = ctrl_angular_speed[-1500:]
                    loop_index = loop_index + 1
                    if loop_index >= data_length:
                        # Draw plot angle and angular control command data
                        if plot_angles:
                            plt.figure(1)
                            plt.plot(ctrl_time, ctrl_angles)
                            plt.legend(['joint' + str(i + 1) for i in range(6)])
                            plt.xlabel('Time(s)')
                            plt.ylabel('Angle(rad)')
                            plt.savefig('../Output/ctrlAngles.png')
                            plt.show()

                            plt.figure(2)
                            plt.plot(ctrl_time, ctrl_angular_speed)
                            plt.legend(['joint' + str(i + 1) for i in range(6)])
                            plt.xlabel('Time(s)')
                            plt.ylabel('AngularSpeed(rad/s)')
                            plt.savefig('../Output/ctrlAngularSpeed.png')
                            plt.show()
                        break
                # Update tqdm bar
                pbar.update()
    finally:
        pybullet.disconnect(simulation.client)
        if disp_human_demonstrate_file_ish5:
            demonstrate_data.close()
