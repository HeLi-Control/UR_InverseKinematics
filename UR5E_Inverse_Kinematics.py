import math

import numpy
import h5py
from scipy.spatial.transform import Rotation
import pybullet

from loguru import logger

from math_utils import (
    get_yzy_euler_angles_from_rotation_matrix,
    unwind_angle_list,
    cvt_target,
    vector_dot_loss,
)
from pybullet_draw_display import (
    set_display_lifetime,
    disp_human_demonstrate,
    draw_coordinate,
)

global_z_offset = 1.0
sleep_time_per_frame = 0.00
calculate_orientation_loss = False
live_plot_display = False
given_fixed_orientation = False
fixed_orientation = [1.0, 0.0, 0.0, 0.0]
draw_end_effector_coordinate = False
disp_human_demonstrate_file = './DemonstrateData/demo.pkl'
# disp_human_demonstrate_file = './DemonstrateData/humanDemonstrate.h5'
disp_human_demonstrate_file_ish5 = disp_human_demonstrate_file.endswith('h5')
disp_debug_params = False


class UR5E_Inverse_Kinematics_Simulation:
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
        self.robot_id = pybullet.loadURDF(fileName=urdf_file, basePosition=[0, 0, global_z_offset], useFixedBase=True)
        # Set camera
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=1.8, cameraYaw=95, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.3 + global_z_offset],
        )
        # Get joints available
        self.all_joints_num = pybullet.getNumJoints(self.robot_id)
        self.end_effector_joint_index = [6]
        self.available_joints_num = len(self.available_joints_indices)

    @property
    def available_joints_indices(self) -> list[int]:
        # [1, 2, 3, 4, 5, 6]
        return [index for index in range(self.all_joints_num) if self.get_joint_type(index) != pybullet.JOINT_FIXED]

    @property
    def available_joint_names(self) -> list[str]:
        # ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        #  'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        return [self.get_joint_name(_joint) for _joint in self.available_joints_indices]

    @property
    def arm_base_position(self) -> list[float]:
        return self.get_link_position_xyz(0)

    @property
    def ee_orientation_quaternion(self) -> list[float]:
        return self.get_link_orientation_quaternion(self.end_effector_joint_index[0])

    def get_joint_name(self, joint_index: int) -> str:
        return str(pybullet.getJointInfo(bodyUniqueId=self.robot_id, jointIndex=joint_index)[1])[2:-1]

    def get_joint_type(self, joint_index: int) -> int:
        return pybullet.getJointInfo(self.robot_id, joint_index)[2]

    def get_joint_angle_rad(self, joint_index: int) -> float:
        return pybullet.getJointState(self.robot_id, joint_index)[0]

    def get_link_position_xyz(self, link_index: int) -> list[float]:
        return list(pybullet.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)[4])

    def get_link_orientation_quaternion(self, link_index: int) -> list[float]:
        return pybullet.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)[5]

    def step_simulation(self, joint_angles: list[float]) -> None:
        if joint_angles:
            pybullet.setJointMotorControlArray(
                bodyUniqueId=self.robot_id, jointIndices=self.available_joints_indices,
                controlMode=pybullet.POSITION_CONTROL, targetPositions=joint_angles,
            )
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING)
        pybullet.stepSimulation(self.client)

    def __calculate_inverse_kinematics_without_orientation(
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

        # Wrist inverse kinematics
        wrist_base_ori = [quaternion_2_numpy_matrix(self.get_link_orientation_quaternion(3))]
        wrist_target_ori = [quaternion_2_numpy_matrix(target_ori) for target_ori in target_orientations]

        def unwind_euler_angle_lists(
                center_angle: list[float], target_angles: list[list[float]]
        ) -> list[list[float]]:
            return [
                unwind_angle_list([0] * len(center_angle), unwind_angle_list(center_angle, target_angle),
                                  period=math.pi * 4, step_size=math.pi * 2)
                for target_angle in target_angles
            ]

        last3dof_ang = [unwind_euler_angle_lists(now_angle[0],
                                                 get_wrist_last3dof(wrist_target_ori[0], wrist_base_ori[0]))]
        last3dof_ang = last3dof_ang[0:1]
        logger.debug(last3dof_ang)
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
            target_orientations: list[list[float]],
    ) -> list[float]:
        now_ang = [self.get_joint_angle_rad(index) for index in self.available_joints_indices]
        # Arm inverse kinematics
        _angles = self.__calculate_inverse_kinematics_without_orientation(
            target_joints_indices=[6, 3],
            target_positions=[target_positions[target_joint] for target_joint in (2, 1)],
        )
        # Wrist orientation inverse kinematics
        ang = self.end_effector_inverse_kinematics_last3dof(
            target_orientations, [now_ang[3:6]], random_select=False
        )
        _angles[3:6] = ang[0]

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


def get_real_target(
        arm_base_position, _target: list[list[float]], man_scale: list
) -> list[list[float]]:
    if not man_scale:
        man_scale = [1.6, 1.5]
    target_points = cvt_target(_target, arm_base_position, _man_scale=man_scale)
    disp_human_demonstrate(target_points, draw_bias=[0, -0.6, global_z_offset])
    pybullet.addUserDebugPoints(
        pointPositions=target_points,
        pointColorsRGB=[[1, 0, 0]] * len(target_points),
        pointSize=2, lifeTime=0.1,
    )
    return target_points


def calculate_orientation_error(
        target_orientation_quaternion: list[list[float]],
        now_orientation_quaternion: list[list[float]],
) -> float:
    error = [
        vector_dot_loss(target_ori, now_ori)
        for target_ori, now_ori in zip(target_orientation_quaternion, now_orientation_quaternion)
    ]
    return numpy.linalg.norm(numpy.array(error)) / math.sqrt(2) / 2


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    if disp_human_demonstrate_file_ish5:
        demonstrate_data = h5py.File(name=disp_human_demonstrate_file, mode="r")
    else:
        demonstrate_data = numpy.load(file=disp_human_demonstrate_file, allow_pickle=True)
    simulation = UR5E_Inverse_Kinematics_Simulation("./RobotDescription/ur5e/ur5e.urdf", disp_debug_params)
    display_button_id = pybullet.addUserDebugParameter(paramName="Display", rangeMin=1, rangeMax=0, startValue=1)
    display_button_id_last_value = 1
    display_demonstrate_flag = True
    demonstrate_scale_id = [
        pybullet.addUserDebugParameter(paramName="Scale1", rangeMin=0.1, rangeMax=10, startValue=1.6),
        pybullet.addUserDebugParameter(paramName="Scale2", rangeMin=0.1, rangeMax=10, startValue=1.5)
    ]
    set_display_lifetime(0.01)

    try:
        ori_err = []
        with tqdm(total=len(list(demonstrate_data["l_arm"])), unit='Frames') as pbar:
            i = 0
            while True:
                if (disp_debug_params and (
                        pybullet.readUserDebugParameter(display_button_id)
                        != display_button_id_last_value)):
                    display_button_id_last_value = display_button_id_last_value + 1
                    display_demonstrate_flag = not display_demonstrate_flag
                if disp_human_demonstrate_file_ish5:
                    target = demonstrate_data["l_arm"][i].tolist() + demonstrate_data["r_arm"][i].tolist()
                else:
                    target = demonstrate_data["l_arm"][i] + demonstrate_data["r_arm"][i]
                if disp_debug_params:
                    target = get_real_target(simulation.arm_base_position, target,
                                             man_scale=[
                                                 pybullet.readUserDebugParameter(demonstrate_scale_id[0]),
                                                 pybullet.readUserDebugParameter(demonstrate_scale_id[1])
                                             ])
                else:
                    target = get_real_target(simulation.arm_base_position, target, [])
                if given_fixed_orientation:
                    angles = simulation.calculate_inverse_kinematics(
                        target_positions=target,
                        target_orientations=[fixed_orientation],
                    )
                    if draw_end_effector_coordinate:
                        simulation.draw_end_effector_coordinate([fixed_orientation])
                    ori_err.append(
                        calculate_orientation_error(
                            [fixed_orientation],
                            [simulation.ee_orientation_quaternion],
                        )
                    )
                else:
                    ori_data = demonstrate_data["ee_ori"][i]
                    if disp_human_demonstrate_file_ish5:
                        angles = simulation.calculate_inverse_kinematics(
                            target_positions=target, target_orientations=ori_data,
                        )
                    else:
                        angles = simulation.calculate_inverse_kinematics(
                            target_positions=target, target_orientations=ori_data,
                        )
                    if draw_end_effector_coordinate:
                        simulation.draw_end_effector_coordinate(ori_data)
                    ori_err.append(
                        calculate_orientation_error(
                            ori_data, [simulation.ee_orientation_quaternion],
                        )
                    )
                simulation.step_simulation(angles)
                pbar.update()
                if calculate_orientation_loss and live_plot_display:
                    plt.clf()
                    plt.plot([i + 1 for i in range(len(ori_err))], ori_err, "b*-")
                    plt.ylabel("orientation error")
                    plt.pause(sleep_time_per_frame)
                    plt.ioff()
                if calculate_orientation_loss and not live_plot_display:
                    plt.plot([i + 1 for i in range(len(ori_err))], ori_err, "b*-")
                    plt.ylabel("orientation error")
                if display_demonstrate_flag:
                    i = i + 1
                    if i >= len(list(demonstrate_data["l_arm"])):
                        break
        if calculate_orientation_loss:
            plt.savefig("./orientation_error.png")
            pybullet.disconnect(simulation.client)
            logger.info(min(ori_err))

    except KeyboardInterrupt:
        pybullet.disconnect(simulation.client)
