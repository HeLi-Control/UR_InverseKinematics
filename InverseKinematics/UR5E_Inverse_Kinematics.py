import math

import numpy
import h5py
from scipy.spatial.transform import Rotation
import pybullet

from tqdm import tqdm

from InverseKinematics.UR5_Inverse_Kinematics import ur5_robot_inverse_kinematics, read_frame_demonstrate_data

from Utils.math_utils import (
    get_yzy_euler_angles_from_rotation_matrix,
    unwind_angle_list,
    cvt_target,
)
from Utils.pybullet_draw_display import (
    set_display_lifetime,
    disp_human_demonstrate_arm,
)

global_z_offset = 1.0
given_fixed_orientation = False
fixed_orientation = [1.0, 0.0, 0.0, 0.0]
draw_end_effector_coordinate = True
calculate_wrist_orientation_self = True


class ur5e_robot_inverse_kinematics(ur5_robot_inverse_kinematics):
    def __init__(self, urdf_file: str, ik_use_world_orientation=True, show_gui=False):
        super(ur5e_robot_inverse_kinematics, self).__init__(urdf_file, ik_use_world_orientation, show_gui)
        self.end_effector_joint_index = [6]

    # .available_joints_indices = [1, 2, 3, 4, 5, 6]
    # .available_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
    #    'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    @property
    def arm_base_position(self) -> list[float]:
        return self.get_link_position_xyz(1)

    @property
    def ee_orientation_quaternion(self) -> list[float]:
        return self.get_link_orientation_quaternion(self.end_effector_joint_index[0])

    def end_effector_inverse_kinematics_last3dof(
            self,
            target_orientations: list[float],
            now_angle: list[list[float]],
            random_select=False
    ) -> list[list[float]]:
        def quaternion_2_numpy_matrix(quaternion: list[float]) -> numpy.matrix:
            return numpy.matrix(numpy.array(Rotation.from_quat(quaternion).as_matrix()))

        def unwind_euler_angle_lists(
                center_angle: list[float], target_angles: list[list[float]]
        ) -> list[list[float]]:
            return [
                unwind_angle_list([0] * len(center_angle), unwind_angle_list(center_angle, target_angle),
                                  period=math.pi * 4, step_size=math.pi * 2)
                for target_angle in target_angles
            ]

        if self.ik_use_world_orientation:
            def get_wrist_last3dof(
                    ee_ori: numpy.matrix,
                    base: numpy.matrix,
                    base_transform=numpy.matrix(numpy.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])),
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
                    quaternion_2_numpy_matrix(target_orientations) @ numpy.matrix(
                        numpy.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))))]
        last3dof_ang = last3dof_ang[0:1]
        # Select the proper result
        if random_select:
            return last3dof_ang[0]
        angle_error = numpy.sum(
            numpy.abs(numpy.array(last3dof_ang) - numpy.array([[ang] for ang in now_angle])), axis=2,
        )
        min_index = numpy.argmin(angle_error, axis=1).tolist()
        return [wrist_angles[index] for wrist_angles, index in zip(last3dof_ang, min_index)]
        # return [wrist_angles[0] for wrist_angles in last3dof_ang]

    def calculate_inverse_kinematics(
            self,
            target_positions: list[list[float]],
            target_orientations: list[float],
            use_elbow_pos=True
    ) -> list[float]:
        now_ang = [self.get_joint_angle_rad(index) for index in self.available_joints_indices]
        # Arm inverse kinematics
        _angles = self.calculate_inverse_kinematics_without_orientation(
            target_joints_indices=[6, 3] if use_elbow_pos else [6],
            target_positions=[target_positions[target_joint] for target_joint in ([2, 1] if use_elbow_pos else [2])])
        # Wrist orientation inverse kinematics
        ang = self.end_effector_inverse_kinematics_last3dof(
            target_orientations, [now_ang[3:6]], random_select=False
        )
        _angles[3:6] = ang[0]

        return unwind_angle_list(
            [0] * len(_angles), unwind_angle_list(now_ang, _angles), period=4 * math.pi, step_size=2 * math.pi,
        )

    def draw_end_effector_coordinate(self, target_orientation: list[float]) -> None:
        self.draw_single_joint_coordinate(self.end_effector_joint_index[0], target_orientation)

    @staticmethod
    def get_real_target(
            arm_base_position: list[float], _target: list[list[float]], man_scale: list
    ) -> list[list[float]]:
        if not man_scale:
            man_scale = [1.4, 1.9]
        target_points = cvt_target(_target, arm_base_position, _man_scale=man_scale)
        disp_human_demonstrate_arm(target_points, draw_bias=[0, -0.6, global_z_offset])
        pybullet.addUserDebugPoints(
            pointPositions=target_points[:3], pointColorsRGB=[[1, 0, 0]] * len(target_points[:3]),
            pointSize=2, lifeTime=0.1,
        )
        return target_points

    def given_demonstrate_data_step_simulation(self, _target_pos: list[list[float]],
                                               _target_ori: list[float]) -> bool:
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
        _target_pos = self.get_real_target(simulation.arm_base_position, _target_pos,
                                           man_scale=interact_scale if self.show_gui else [])
        # Inverse Kinematics
        angles = simulation.calculate_inverse_kinematics(_target_pos, _target_ori)
        # Step simulation
        simulation.step_simulation(angles)
        return self.display_demonstrate_flag if self.show_gui else True


if __name__ == "__main__":
    disp_human_demonstrate_file = '../DemonstrateData/demo.pkl'
    # disp_human_demonstrate_file = '../DemonstrateData/humanDemonstrate.h5'
    disp_human_demonstrate_file_ish5 = disp_human_demonstrate_file.endswith('h5')
    # Load demonstrate data
    demonstrate_data = h5py.File(name=disp_human_demonstrate_file, mode="r") if disp_human_demonstrate_file_ish5 \
        else numpy.load(file=disp_human_demonstrate_file, allow_pickle=True)
    # Start simulation
    simulation = ur5e_robot_inverse_kinematics(urdf_file="../RobotDescription/ur5e/ur5e.urdf",
                                               ik_use_world_orientation=True, show_gui=False)
    set_display_lifetime(0.01)
    # Simulation in loop
    try:
        with (tqdm(total=len(list(demonstrate_data["l_arm"])), unit='Frames') as pbar):
            loop_index = 0
            while True:
                # Read demonstrate data
                target_pos, target_ori = read_frame_demonstrate_data(
                    demonstrate_data, loop_index, calculate_wrist_orientation_self, disp_human_demonstrate_file_ish5,
                    given_fixed_orientation, fixed_orientation)
                # WARNING: Negative Position Z in both position and orientation
                for i in range(3):
                    target_pos[i][2] = -target_pos[i][2]
                target_ori = [list(Rotation.from_matrix(
                    Rotation.from_quat(target_ori[0]).as_matrix() @
                    numpy.matrix(numpy.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))).as_quat(canonical=True))]
                # Draw orientation coordinate
                if draw_end_effector_coordinate:
                    simulation.draw_end_effector_coordinate(target_ori[0])
                # Step simulation
                if simulation.given_demonstrate_data_step_simulation(target_pos, target_ori[0]):
                    loop_index = loop_index + 1
                    if loop_index >= len(list(demonstrate_data["l_arm"])):
                        break
                # Update tqdm bar
                pbar.update()
    except KeyboardInterrupt:
        pybullet.disconnect(simulation.client)
