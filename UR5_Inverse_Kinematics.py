import math

import numpy
import h5py
from scipy.spatial.transform import Rotation
import pybullet

from math_utils import (
    unwind_angles,
    unwind_angle_list,
    point_transfer_scale,
    vector_dot_loss,
)
from pybullet_draw_display import (
    set_display_lifetime,
    disp_human_demonstrate,
    draw_coordinate,
)

global_z_offset = 1.0
sleep_time_per_frame = 0.001
calculate_orientation_loss = True
live_plot_display = True
given_fixed_orientation = False
fixed_orientation = [[0, 1, 0, 0], [0, 0, 1, 0]]
draw_end_effector_coordinate = False


class UR5_Inverse_Kinematics_Simulation:
    def __init__(self, urdf_file: str, show_gui=False):
        # Connect the client
        self.client = pybullet.connect(pybullet.GUI)
        if show_gui:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1)
        else:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 0)
        # Add source path
        import pybullet_data

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Load land
        pybullet.loadURDF("plane.urdf")
        # Load robot
        self.robot_id = pybullet.loadURDF(
            fileName=urdf_file, basePosition=[0, 0, global_z_offset]
        )
        # Set camera
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=95,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.5 + global_z_offset],
        )
        # Get joints available
        self.all_joints_num = pybullet.getNumJoints(self.robot_id)
        self.end_effector_joint_index = (7, 16)
        self.available_joints_num = len(self.available_joints_indices)

    @property
    def available_joints_indices(self) -> list[int]:
        # [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16]
        return [
            index
            for index in range(self.all_joints_num)
            if self.get_joint_type(index) != pybullet.JOINT_FIXED
        ]

    @property
    def available_joint_names(self) -> list[str]:
        # ['left_shoulder_pan_joint', 'left_shoulder_lift_joint', 'left_elbow_joint',
        #  'left_wrist_1_joint', 'left_wrist_2_joint', 'left_wrist_3_joint',
        #  'right_shoulder_pan_joint', 'right_shoulder_lift_joint', 'right_elbow_joint',
        #  'right_wrist_1_joint', 'right_wrist_2_joint', 'right_wrist_3_joint']
        return [self.get_joint_name(_joint) for _joint in self.available_joints_indices]

    @property
    def arm_base_position(self) -> tuple[list[float], list[float]]:
        return self.get_link_position_xyz(2), self.get_link_position_xyz(11)

    @property
    def ee_orientation_quaternion(self) -> list[list[float]]:
        return [
            list(self.get_link_orientation_quaternion(ee))
            for ee in self.end_effector_joint_index
        ]

    def get_joint_name(self, joint_index: int) -> str:
        return str(
            pybullet.getJointInfo(bodyUniqueId=self.robot_id, jointIndex=joint_index)[1]
        )[2:-1]

    def get_joint_type(self, joint_index: int) -> int:
        return pybullet.getJointInfo(self.robot_id, joint_index)[2]

    def get_joint_angle_rad(self, joint_index: int) -> float:
        return pybullet.getJointState(self.robot_id, joint_index)[0]

    def get_link_position_xyz(self, link_index: int) -> list[float]:
        return list(
            pybullet.getLinkState(
                self.robot_id, link_index, computeForwardKinematics=True
            )[4]
        )

    def get_link_orientation_quaternion(self, link_index: int) -> list[float]:
        return pybullet.getLinkState(
            self.robot_id, link_index, computeForwardKinematics=True
        )[5]

    def step_simulation(self, joint_angles: list[float]) -> None:
        if joint_angles:
            pybullet.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.available_joints_indices,
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=joint_angles,
            )
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
        pybullet.stepSimulation(self.client)

    def __calculate_inverse_kinematics_without_orientation(
        self, target_joints_indices: list[float], target_positions: list[list]
    ) -> list[float]:
        return list(
            pybullet.calculateInverseKinematics2(
                bodyUniqueId=self.robot_id,
                endEffectorLinkIndices=target_joints_indices,
                targetPositions=target_positions,
            )
        )

    def end_effector_inverse_kinematics_last3dof(
        self,
        target_orientations: list[list[float]],
        now_angle: list[list[float]],
        random_select=False,
    ) -> list[list[float]]:
        def quaternion_2_numpy_matrix(quaternion: list[float]) -> numpy.matrix:
            return numpy.matrix(numpy.array(Rotation.from_quat(quaternion).as_matrix()))

        def get_yzy_euler_angles_from_rotation_matrix(
            rotation_matrix: numpy.matrix,
        ) -> list[list[float]]:
            ret_angles = (
                Rotation.from_matrix(rotation_matrix).as_euler(seq="yzy", degrees=False)
                * -1
            )
            euler_angle = [unwind_angle_list([0] * numpy.size(ret_angles), ret_angles.tolist()) for _ in range(3)]

            euler_angle[1][0] = euler_angle[1][0] - math.pi
            euler_angle[1][1] = -euler_angle[1][1]
            euler_angle[1][2] = euler_angle[1][2] - math.pi
            euler_angle[1] = [unwind_angles(0, ang) for ang in euler_angle[1]]

            euler_angle[2][0] = math.pi - euler_angle[2][0]
            euler_angle[2][2] = math.pi - euler_angle[2][2]
            euler_angle[2] = [unwind_angles(0, ang) for ang in euler_angle[2]]

            return euler_angle

        def get_wrist_last3dof(
            ee_ori: numpy.matrix,
            base: numpy.matrix,
            base_transform=numpy.matrix(
                numpy.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            ),
        ) -> list[list[float]]:
            rotation_matrix = ee_ori.transpose() @ base @ base_transform
            return get_yzy_euler_angles_from_rotation_matrix(rotation_matrix)

        # Wrist inverse kinematics
        wrist_base_ori = [
            quaternion_2_numpy_matrix(self.get_link_orientation_quaternion(_i))
            for _i in [4, 13]
        ]
        wrist_target_ori = [
            quaternion_2_numpy_matrix(target_ori) for target_ori in target_orientations
        ]
        # Select the proper result
        if random_select:
            return [
                get_wrist_last3dof(wrist_target_ori[_i], wrist_base_ori[_i])[0]
                for _i in [0, 1]
            ]

        def unwind_euler_angle_lists(
            center_angle: list[float], target_angles: list[list[float]]
        ) -> list[list[float]]:
            return [
                unwind_angle_list(
                    [0] * len(center_angle),
                    unwind_angle_list(center_angle, target_angle),
                    period=4 * math.pi,
                    step_size=2 * math.pi,
                )
                for target_angle in target_angles
            ]

        last3dof_ang = [
            unwind_euler_angle_lists(now_angle[_i], get_wrist_last3dof(wrist_target_ori[_i], wrist_base_ori[_i]))
            for _i in [0, 1]
        ]
        angle_error = numpy.sum(
            numpy.abs(
                numpy.array(last3dof_ang) - numpy.array([[ang] for ang in now_angle])
            ),
            axis=2,
        )
        min_index = numpy.argmin(angle_error, axis=1).tolist()
        return [wrist_angles[index] for wrist_angles, index in zip(last3dof_ang, min_index)]
        # return [wrist_angles[0] for wrist_angles in last3dof_ang]

    def calculate_inverse_kinematics(
        self,
        target_positions: list[list[float]],
        target_orientations: list[list[float]],
    ) -> list[float]:
        now_ang = [
            self.get_joint_angle_rad(index) for index in self.available_joints_indices
        ]
        # Arm inverse kinematics
        _angles = self.__calculate_inverse_kinematics_without_orientation(
            target_joints_indices=[3, 5, 7, 12, 14, 16],
            target_positions=target_positions,
        )
        # Wrist orientation inverse kinematics
        ang = self.end_effector_inverse_kinematics_last3dof(
            target_orientations, [now_ang[3:6], now_ang[9:12]], random_select=True
        )
        _angles[3:6] = ang[0]
        _angles[9:12] = ang[1]

        return unwind_angle_list(
            [0] * len(_angles),
            unwind_angle_list(now_ang, _angles),
            period=4 * math.pi,
            step_size=2 * math.pi,
        )

    def draw_end_effector_coordinate(
        self, target_orientations: list[list[float]]
    ) -> None:
        for index in range(len(self.end_effector_joint_index)):
            draw_coordinate(
                self.get_link_position_xyz(self.end_effector_joint_index[index]),
                target_orientations[index],
            )
            draw_coordinate(
                self.get_link_position_xyz(self.end_effector_joint_index[index]),
                self.get_link_orientation_quaternion(
                    self.end_effector_joint_index[index]
                ),
            )


def get_real_target(
    arm_base_position: tuple, _target: list[list[float]]
) -> list[list[float]]:
    disp_human_demonstrate(_target, [0, -0.6, 1.5 + global_z_offset])

    def cvt_target(
        target_point: list[list[float]],
        left_base_pos: list[float],
        right_base_pos: list[float],
        man_scale=2.2,
    ) -> list[list[float]]:
        scale = [man_scale, man_scale]
        return point_transfer_scale(
            target_point[:3], target_point[0], left_base_pos, scale=scale[0]
        ) + point_transfer_scale(
            target_point[3:], target_point[3], right_base_pos, scale=scale[1]
        )

    target_points = cvt_target(_target, *arm_base_position)
    pybullet.addUserDebugPoints(
        pointPositions=target_points,
        pointColorsRGB=[[1, 0, 0]] * len(target_points),
        pointSize=2,
        lifeTime=0.1,
    )
    return target_points


def calculate_orientation_error(
    target_orientation_quaternion: list[list[float]],
    now_orientation_quaternion: list[list[float]],
) -> float:
    error = [
        vector_dot_loss(target_ori, now_ori)
        for target_ori, now_ori in zip(
            target_orientation_quaternion, now_orientation_quaternion
        )
    ]
    return numpy.linalg.norm(numpy.array(error)) / math.sqrt(2) / 2


if __name__ == "__main__":
    from tqdm import trange
    import matplotlib.pyplot as plt

    demonstrate_file = h5py.File(name="./humanDemonstrate.h5", mode="r")
    simulation = UR5_Inverse_Kinematics_Simulation(
        "./ur_description/ur5_robot_hand.urdf"
    )
    set_display_lifetime(0.01)
    try:
        ori_err = []
        for i in trange(len(list(demonstrate_file["l_arm"]))):
            for _ in range(3):
                target = (
                    demonstrate_file["l_arm"][i].tolist()
                    + demonstrate_file["r_arm"][i].tolist()
                )
                target = get_real_target(simulation.arm_base_position, target)
                if given_fixed_orientation:
                    angles = simulation.calculate_inverse_kinematics(
                        target_positions=target,
                        target_orientations=fixed_orientation,
                    )
                    if draw_end_effector_coordinate:
                        simulation.draw_end_effector_coordinate(fixed_orientation)
                    ori_err.append(
                        calculate_orientation_error(
                            fixed_orientation,
                            simulation.ee_orientation_quaternion,
                        )
                    )
                else:
                    angles = simulation.calculate_inverse_kinematics(
                        target_positions=target,
                        target_orientations=demonstrate_file["ee_ori"][i].tolist(),
                    )
                    if draw_end_effector_coordinate:
                        simulation.draw_end_effector_coordinate(
                            demonstrate_file["ee_ori"][i].tolist()
                        )
                    ori_err.append(
                        calculate_orientation_error(
                            demonstrate_file["ee_ori"][i].tolist(),
                            simulation.ee_orientation_quaternion,
                        )
                    )
                simulation.step_simulation(angles)
            if calculate_orientation_loss and live_plot_display:
                plt.clf()
                plt.plot([i + 1 for i in range(len(ori_err))], ori_err, "b*-")
                plt.ylabel("orientation error")
                plt.pause(sleep_time_per_frame)
                plt.ioff()
        if calculate_orientation_loss and not live_plot_display:
            plt.plot([i + 1 for i in range(len(ori_err))], ori_err, "b*-")
            plt.ylabel("orientation error")
        if calculate_orientation_loss:
            plt.savefig("./orientation_error.png")
            pybullet.disconnect(simulation.client)
            print(min(ori_err))

    except KeyboardInterrupt:
        pybullet.disconnect(simulation.client)
