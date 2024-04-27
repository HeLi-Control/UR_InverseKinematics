import math

import numpy
import h5py
from scipy.spatial.transform import Rotation
import pybullet

from tqdm import tqdm

from Utils.math_utils import (
    get_yzy_euler_angles_from_rotation_matrix,
    unwind_angle_list,
    cvt_target_bimanual,
)
from Utils.pybullet_draw_display import (
    set_display_lifetime,
    disp_human_demonstrate_bimanual_arm,
    draw_coordinate,
)

global_z_offset = 1.0
given_fixed_orientation = False
fixed_orientation = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
draw_end_effector_coordinate = False


class ur5_robot_inverse_kinematics:
    def __init__(self, urdf_file: str, show_gui=False):
        # Connect the client
        self.client = pybullet.connect(pybullet.GUI)
        self.show_gui = show_gui
        if show_gui:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1)
            # Initialize debug parameters
            self.display_button_id = pybullet.addUserDebugParameter(paramName="Display", rangeMin=1, rangeMax=0,
                                                                    startValue=1)
            self.display_button_id_last_value = 1
            self.display_demonstrate_flag = True
            self.demonstrate_scale_id = [
                pybullet.addUserDebugParameter(paramName="Scale1", rangeMin=0.1, rangeMax=10, startValue=1.6),
                pybullet.addUserDebugParameter(paramName="Scale2", rangeMin=0.1, rangeMax=10, startValue=1.5)
            ]
        else:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 0)
        # Add source path
        import pybullet_data
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Load land
        pybullet.loadURDF("plane.urdf")
        # Load robot
        self.robot_id = pybullet.loadURDF(fileName=urdf_file, basePosition=[0, 0, global_z_offset], useFixedBase=True)
        # Set camera
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=1.8, cameraYaw=95, cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.5 + global_z_offset],
        )
        # Get joints available
        self.all_joints_num = pybullet.getNumJoints(self.robot_id)
        self.end_effector_joint_index = [7, 16]
        self.available_joints_num = len(self.available_joints_indices)

    @property
    def available_joints_indices(self) -> list[int]:
        # [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16]
        return [index for index in range(self.all_joints_num) if self.get_joint_type(index) != pybullet.JOINT_FIXED]

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
        return [list(self.get_link_orientation_quaternion(ee)) for ee in self.end_effector_joint_index]

    def get_joint_name(self, joint_index: int) -> str:
        return str(pybullet.getJointInfo(bodyUniqueId=self.robot_id, jointIndex=joint_index)[1])[2:-1]

    def get_joint_type(self, joint_index: int) -> int:
        return pybullet.getJointInfo(self.robot_id, joint_index)[2]

    def get_joint_angle_rad(self, joint_index: int) -> float:
        return pybullet.getJointState(self.robot_id, joint_index)[0]

    def get_link_position_xyz(self, link_index: int) -> list[float]:
        return list(pybullet.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)[4])

    def get_link_orientation_quaternion(self, link_index: int) -> list[float]:
        return list(pybullet.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)[5])

    def step_simulation(self, joint_angles: list[float]) -> None:
        if joint_angles:
            pybullet.setJointMotorControlArray(
                bodyUniqueId=self.robot_id, jointIndices=self.available_joints_indices,
                controlMode=pybullet.POSITION_CONTROL, targetPositions=joint_angles,
            )
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
        pybullet.stepSimulation(self.client)

    def calculate_inverse_kinematics_without_orientation(
            self, target_joints_indices: list[float], target_positions: list[list]
    ) -> list[float]:
        return list(
            pybullet.calculateInverseKinematics2(
                bodyUniqueId=self.robot_id, endEffectorLinkIndices=target_joints_indices,
                targetPositions=target_positions,
            )
        )

    def end_effector_inverse_kinematics_last3dof(
            self,
            target_orientations: list[list[float]],
            now_angle: list[list[float]],
            random_select=False,
            orientation_world=True
    ) -> list[list[float]]:
        def quaternion_2_numpy_matrix(quaternion: list[float]) -> numpy.matrix:
            return numpy.matrix(numpy.array(Rotation.from_quat(quaternion).as_matrix()))

        def get_wrist_last3dof(
                ee_ori: numpy.matrix,
                base: numpy.matrix,
                base_transform=numpy.matrix(
                    numpy.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                ),
        ) -> list[list[float]]:
            rotation_matrix = ee_ori.transpose() @ base @ base_transform
            return get_yzy_euler_angles_from_rotation_matrix(rotation_matrix)

        def unwind_euler_angle_lists(
                center_angle: list[float], target_angles: list[list[float]]
        ) -> list[list[float]]:
            return [
                unwind_angle_list(
                    [0] * len(center_angle), unwind_angle_list(center_angle, target_angle),
                    period=4 * math.pi, step_size=2 * math.pi,
                )
                for target_angle in target_angles
            ]

        if orientation_world:
            # Wrist inverse kinematics
            wrist_base_ori = [quaternion_2_numpy_matrix(self.get_link_orientation_quaternion(_i)) for _i in [4, 13]]
            wrist_target_ori = [quaternion_2_numpy_matrix(target_orientation) for target_orientation in
                                target_orientations]
            last3dof_ang = [
                unwind_euler_angle_lists(now_angle[_i], get_wrist_last3dof(wrist_target_ori[_i], wrist_base_ori[_i]))
                for _i in [0, 1]
            ]
        else:
            last3dof_ang = [unwind_euler_angle_lists(
                now_angle[0], get_yzy_euler_angles_from_rotation_matrix(
                    quaternion_2_numpy_matrix(target_orientation) @ numpy.matrix(
                        numpy.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])))) for target_orientation in
                target_orientations]
        # Select the proper result
        if random_select:
            return [last3dof_ang[0][0], last3dof_ang[1][0]]
        angle_error = numpy.sum(
            numpy.abs(numpy.array(last3dof_ang) - numpy.array([[ang] for ang in now_angle])), axis=2,
        )
        min_index = numpy.argmin(angle_error, axis=1).tolist()
        return [wrist_angles[index] for wrist_angles, index in zip(last3dof_ang, min_index)]
        # return [wrist_angles[0] for wrist_angles in last3dof_ang]

    def calculate_inverse_kinematics(
            self,
            target_positions: list[list[float]],
            target_orientations: list[list[float]],
            use_elbow_pos=True
    ) -> list[float]:
        now_ang = [self.get_joint_angle_rad(index) for index in self.available_joints_indices]
        # Arm inverse kinematics
        _angles = self.calculate_inverse_kinematics_without_orientation(
            target_joints_indices=[7, 5, 16, 14] if use_elbow_pos else [7, 16],
            target_positions=[target_positions[target_joint] for target_joint in
                              ([2, 1, 5, 4] if use_elbow_pos else [2, 5])])
        # Wrist orientation inverse kinematics
        ang = self.end_effector_inverse_kinematics_last3dof(
            target_orientations, [now_ang[3:6], now_ang[9:12]], random_select=True, orientation_world=False
        )
        _angles[3:6] = ang[0]
        _angles[9:12] = ang[1]

        return unwind_angle_list(
            [0] * len(_angles), unwind_angle_list(now_ang, _angles), period=4 * math.pi, step_size=2 * math.pi,
        )

    def draw_end_effector_coordinate(
            self, target_orientations: list[list[float]]
    ) -> None:
        for index in range(len(self.end_effector_joint_index)):
            draw_coordinate(
                self.get_link_position_xyz(self.end_effector_joint_index[index]), target_orientations[index],
            )
            draw_coordinate(
                self.get_link_position_xyz(self.end_effector_joint_index[index]),
                self.get_link_orientation_quaternion(self.end_effector_joint_index[index]),
            )

    @staticmethod
    def get_real_target(
            arm_base_position: tuple, _target: list[list[float]], man_scale: list
    ) -> list[list[float]]:
        if not man_scale:
            man_scale = [1.6, 1.5]
            # man_scale = [2.5, 2.5]
        target_points = cvt_target_bimanual(_target, *arm_base_position, _man_scale=man_scale)
        disp_human_demonstrate_bimanual_arm(target_points, draw_bias=[0, -0.6, global_z_offset])
        pybullet.addUserDebugPoints(
            pointPositions=target_points, pointColorsRGB=[[1, 0, 0]] * len(target_points), pointSize=2, lifeTime=0.1,
        )
        return target_points

    def given_demonstrate_data_step_simulation(self, _target_pos: list[list[float]],
                                               _target_ori: list[list[float]]) -> bool:
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
        angles = simulation.calculate_inverse_kinematics(_target_pos, _target_ori, use_elbow_pos=False)
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
    simulation = ur5_robot_inverse_kinematics(urdf_file="../RobotDescription/ur_description/ur5_robot_hand.urdf",
                                              show_gui=False)
    set_display_lifetime(0.01)
    # Simulation in loop
    try:
        with (tqdm(total=len(list(demonstrate_data["l_arm"])), unit='Frames') as pbar):
            loop_index = 0
            while True:
                # Read demonstrate data
                target_pos = demonstrate_data["l_arm"][loop_index].tolist() + demonstrate_data["r_arm"][
                    loop_index].tolist() if disp_human_demonstrate_file_ish5 \
                    else demonstrate_data["l_arm"][loop_index] + demonstrate_data["r_arm"][loop_index]
                target_ori = fixed_orientation if given_fixed_orientation else demonstrate_data["ee_ori"][
                    loop_index]
                # Draw orientation coordinate
                if draw_end_effector_coordinate:
                    simulation.draw_end_effector_coordinate(target_ori)
                # Step simulation
                if simulation.given_demonstrate_data_step_simulation(target_pos, target_ori):
                    loop_index = loop_index + 1
                    if loop_index >= len(list(demonstrate_data["l_arm"])):
                        break
                # Update tqdm bar
                pbar.update()
    except KeyboardInterrupt:
        pybullet.disconnect(simulation.client)
