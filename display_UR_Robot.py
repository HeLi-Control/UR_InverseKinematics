import math

import pybullet
import h5py
from tqdm import trange
import torch
from ikpy.chain import Chain
import xml.etree.ElementTree
import numpy
from scipy.spatial.transform import Rotation
from ur_analytic_ik import ur5e

global_z_offset = 1.0


class Display_Pybullet_UR_Robot:
    def __init__(self):
        # Connect the client
        self.client = pybullet.connect(pybullet.GUI)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        # Add source path
        import pybullet_data

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Load land
        pybullet.loadURDF("plane.urdf")
        # Load robot
        self.robot_id = pybullet.loadURDF(
            fileName="ur_description/ur5_robot_hand.urdf",
            basePosition=[0, 0, global_z_offset],
        )
        # Set camera
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=95,
            cameraPitch=-20,
            cameraTargetPosition=[0, -0.5, 0.5 + global_z_offset],
        )
        # Get joints available
        self.all_joints_num = pybullet.getNumJoints(self.robot_id)
        self.end_effector_joint_index = (7, 16)
        self.available_joints_num = len(self.available_joints_indices)
        # Get joint limits
        self.limit = self.__get_joint_limit__()
        # chain for inverse kinematics
        self.chain_left = Chain.from_urdf_file(urdf_file="ur_description/ur5_robot_hand.urdf",
                                               base_elements=["left_base_link"],
                                               active_links_mask=[False if i in [0, 7] else True for i in range(8)])
        self.chain_right = Chain.from_urdf_file(urdf_file="ur_description/ur5_robot_hand.urdf",
                                                base_elements=["right_base_link"],
                                                active_links_mask=[False if i in [0, 7] else True for i in range(8)])

    @property
    def available_joints_indices(self) -> list[int]:
        # [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16]
        return [
            i
            for i in range(self.all_joints_num)
            if pybullet.getJointInfo(bodyUniqueId=self.robot_id, jointIndex=i)[2]
               != pybullet.JOINT_FIXED
        ]

    @property
    def available_joint_names(self) -> list[str]:
        # ['left_shoulder_pan_joint', 'left_shoulder_lift_joint', 'left_elbow_joint',
        #  'left_wrist_1_joint', 'left_wrist_2_joint', 'left_wrist_3_joint',
        #  'right_shoulder_pan_joint', 'right_shoulder_lift_joint', 'right_elbow_joint',
        #  'right_wrist_1_joint', 'right_wrist_2_joint', 'right_wrist_3_joint']
        return [
            str(
                pybullet.getJointInfo(bodyUniqueId=self.robot_id, jointIndex=_joint)[1]
            )[2:-1]
            for _joint in self.available_joints_indices
        ]

    def __get_joint_limit__(self) -> dict:
        tree = xml.etree.ElementTree.parse("ur_description/ur5_robot_hand.urdf")
        robot = tree.getroot()
        joints = robot.findall("joint")
        import math

        limit = {
            "lower": [-math.inf for _ in range(self.available_joints_num)],
            "upper": [math.inf for _ in range(self.available_joints_num)],
        }
        for joint in joints:
            if joint.get("type") == "revolute":
                if joint.get("name") in self.available_joint_names:
                    lim = joint.find("limit")
                    limit["lower"][
                        self.available_joint_names.index(joint.get("name"))
                    ] = lim.get("lower")
                    limit["upper"][
                        self.available_joint_names.index(joint.get("name"))
                    ] = lim.get("upper")
        return limit

    def step_simulation(self, joint_angles: list[float], joint_angles1=None) -> None:
        if joint_angles:
            pybullet.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.available_joints_indices,
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=joint_angles,
            )
        pybullet.stepSimulation(self.client)

    def calculate_inverse_kinematics_without_orientation(
            self, joints_indices: list[float], target_positions: list[list]
    ) -> list[float]:
        return list(
            pybullet.calculateInverseKinematics2(
                bodyUniqueId=self.robot_id,
                endEffectorLinkIndices=joints_indices,
                targetPositions=target_positions,
            )
        )

    def calculate_inverse_kinematics_given_orientation(
            self,
            target_positions: list[list[float]],
            target_orientations: list[list[float]],
            rest_poses=None,
    ) -> list[float]:
        res = []
        for i in range(2):
            res.append(
                list(
                    pybullet.calculateInverseKinematics(
                        bodyUniqueId=self.robot_id,
                        endEffectorLinkIndex=self.end_effector_joint_index[i],
                        targetPosition=target_positions[i],
                        targetOrientation=target_orientations[i],
                        # lowerLimits=self.limit["lower"],
                        # upperLimits=self.limit["upper"],
                        # jointRanges=[3, 2, 1, 3, 5.8, 4, 6, 3, 2, 1, 3, 5.8, 4, 6],
                        # restPoses=rest_poses,
                    )
                )
            )
        return (
                res[0][: int(self.available_joints_num / 2)]
                + res[1][int(self.available_joints_num / 2):]
        )

    def calc_inverse_kinematics(
            self,
            target_positions: list[list[float]],
            target_orientations: list[list[float]],
    ) -> list[float]:
        _angles = self.calculate_inverse_kinematics_without_orientation(
            joints_indices=[3, 5, 7, 12, 14, 16], target_positions=target_positions
        )
        wrist_now_ori = [numpy.matrix(numpy.array(Rotation.from_quat(pybullet.getLinkState(self.robot_id, i, computeForwardKinematics=True)[5]).as_matrix())) for i in [4, 13]]
        wrist_target_ori = [numpy.matrix(numpy.array(Rotation.from_quat(target_orientations[i]).as_matrix())) for i in [0, 1]]
        wrist_action_ori = [wrist_target_ori[i] @ wrist_now_ori[i].transpose() for i in [0, 1]]
        ang = [Rotation.from_matrix(wrist_action_ori[i]).as_euler(seq='yxy', degrees=False) for i in [0, 1]]
        _angles[3:6] = [-ang for ang in ang[0]]
        _angles[9:12] = [+ang for ang in ang[1]]

        now_ang = [pybullet.getJointState(self.robot_id, index)[0] for index in self.available_joints_indices]
        _angles = [unwind_joints(now_ang[i], _angles[i]) for i in range(len(_angles))]

        # _angles = self.calculate_inverse_kinematics_given_orientation(
        #     target_positions=[target_positions[2], target_positions[5]],
        #     target_orientations=target_orientations,
        #     # rest_poses=[0 for _ in range(12)],
        # )
        return _angles

    def ikpy_calculate_inverse_kinematics(
            self, target_positions: list[list[float]], target_orientations: list[list[float]]
    ) -> list[float]:
        # coordinate transfer
        origin_left = pybullet.getLinkState(self.robot_id, 1, computeForwardKinematics=True)
        origin_right = pybullet.getLinkState(self.robot_id, 10, computeForwardKinematics=True)
        world_target_transfer = [pack_homogeneous_transfer_matrix(target_positions[2], target_orientations[0]),
                                 pack_homogeneous_transfer_matrix(target_positions[5], target_orientations[1])]
        world_origin_transfer = [pack_homogeneous_transfer_matrix(origin_left[4], origin_left[5]),
                                 pack_homogeneous_transfer_matrix(origin_right[4], origin_right[5])]
        origin_target_transfer = [(numpy.matrix(numpy.array(world_target_transfer[0])) @ numpy.matrix(numpy.array(world_origin_transfer[0])).I).tolist(),
                                  (numpy.matrix(numpy.array(world_target_transfer[1])) @ numpy.matrix(numpy.array(world_origin_transfer[1])).I).tolist()]
        target_pos_left, target_ori_left = unpack_homogeneous_transfer_matrix(origin_target_transfer[0])
        target_pos_right, target_ori_right = unpack_homogeneous_transfer_matrix(origin_target_transfer[1])
        pybullet.addUserDebugPoints(
            pointPositions=[target_pos_left],
            pointColorsRGB=[[0, 1, 0]],
            pointSize=10,
            lifeTime=0.1,
        )
        # inverse kinematics
        solutions_left = ur5e.inverse_kinematics(numpy.array(origin_target_transfer[0]))
        solutions_right = ur5e.inverse_kinematics(numpy.array(origin_target_transfer[1]))
        ang_left = self.chain_left.inverse_kinematics(target_position=target_pos_left,
                                                      target_orientation=target_ori_left,
                                                      orientation_mode="all")
        ang_right = self.chain_right.inverse_kinematics(target_position=target_pos_right,
                                                        target_orientation=target_ori_right,
                                                        orientation_mode="all")
        return solutions_left[0].tolist()[0] + solutions_right[0].tolist()[0]


def cvt_point(
        target: list[list[float]], zero_point: list[float], bias: list[float], scale=1.0
) -> list[list[float]]:
    converted_target = (torch.tensor(target) - torch.tensor(zero_point)) * scale + torch.tensor(bias)
    return converted_target.tolist()


def calc_given_arm_len(target: list[list[float]]) -> list[float]:
    target = torch.tensor(target)
    lengths = torch.stack(
        [
            target[1] - target[0],
            target[2] - target[1],
            target[4] - target[3],
            target[5] - target[4],
        ]
    )
    lengths = torch.norm(lengths, dim=1)
    return [float(lengths[0] + lengths[1]), float(lengths[2] + lengths[3])]


def get_target(
        target: list[list[float]], left_base_pos: list[float], right_base_pos: list[float]
) -> list[list[float]]:
    # sum_length = calc_given_arm_len(target)
    man_scale = 2.5
    # scale = [0.53 / sum_length[0] * 1.3, 0.53 / sum_length[1] * man_scale]
    scale = [man_scale, man_scale]
    return (cvt_point(target[:3], target[0], left_base_pos, scale=scale[0]) +
            cvt_point(target[3:], target[3], right_base_pos, scale=scale[1]))


def disp_human_demonstrate(target: list[list[float]],
                           draw_bias=None
                           ) -> None:
    if draw_bias is None:
        draw_bias = [0, -0.6, 1.5 + global_z_offset]
    target = cvt_point(target, [0, 0, 0], draw_bias)
    color = [1, 0, 1]
    lifeTime = 0.08
    pybullet.addUserDebugLine(
        lineFromXYZ=target[0],
        lineToXYZ=target[1],
        lineColorRGB=color,
        lineWidth=4,
        lifeTime=lifeTime,
    )
    pybullet.addUserDebugLine(
        lineFromXYZ=target[1],
        lineToXYZ=target[2],
        lineColorRGB=color,
        lineWidth=4,
        lifeTime=lifeTime,
    )
    pybullet.addUserDebugLine(
        lineFromXYZ=target[3],
        lineToXYZ=target[4],
        lineColorRGB=color,
        lineWidth=4,
        lifeTime=lifeTime,
    )
    pybullet.addUserDebugLine(
        lineFromXYZ=target[4],
        lineToXYZ=target[5],
        lineColorRGB=color,
        lineWidth=4,
        lifeTime=lifeTime,
    )


def draw_coordinate(position: list[float], orientation: list[float]) -> None:
    pybullet.addUserDebugLine(lineFromXYZ=position,
                              lineToXYZ=(Rotation.from_quat(orientation).as_matrix()[:, 0].squeeze() + numpy.array(
                                  position)).tolist(),
                              lineColorRGB=[1, 0, 0], lineWidth=4, lifeTime=0.1)
    pybullet.addUserDebugLine(lineFromXYZ=position,
                              lineToXYZ=(Rotation.from_quat(orientation).as_matrix()[:, 1].squeeze() + numpy.array(
                                  position)).tolist(),
                              lineColorRGB=[0, 1, 0], lineWidth=4, lifeTime=0.1)
    pybullet.addUserDebugLine(lineFromXYZ=position,
                              lineToXYZ=(Rotation.from_quat(orientation).as_matrix()[:, 2].squeeze() + numpy.array(
                                  position)).tolist(),
                              lineColorRGB=[0, 0, 1], lineWidth=4, lifeTime=0.1)


def pack_homogeneous_transfer_matrix(translate: list[float], rotation: list[float]) -> list[list[float]]:
    return numpy.vstack(
        (numpy.hstack((Rotation.from_quat(rotation).as_matrix(), numpy.array(translate).reshape(-1, 1))),
         numpy.array([0, 0, 0, 1]))).tolist()


def unpack_homogeneous_transfer_matrix(homogeneous_transfer_matrix: list[list[float]]) -> (list[float], list[list[float]]):
    return (numpy.array(homogeneous_transfer_matrix)[:3, 3].tolist(),
            numpy.array(homogeneous_transfer_matrix)[:3, :3].tolist())

def unwind_joints(now_angle: float, target_angle: float) -> float:
    while target_angle - now_angle > math.pi:
        target_angle = target_angle - 2 * math.pi
    while target_angle - now_angle < -math.pi:
        target_angle = target_angle + 2 * math.pi
    return target_angle



if __name__ == "__main__":
    simulation = Display_Pybullet_UR_Robot()
    demonstrate_file = h5py.File(name="humanDemonstrate.h5", mode="r")
    for i in trange(len(list(demonstrate_file["l_arm"]))):
        target = demonstrate_file["l_arm"][i].tolist() + demonstrate_file["r_arm"][i].tolist()
        disp_human_demonstrate(target)
        target_points = get_target(
            target=target,
            left_base_pos=pybullet.getLinkState(
                simulation.robot_id, 2, computeForwardKinematics=1
            )[0],
            right_base_pos=pybullet.getLinkState(
                simulation.robot_id, 11, computeForwardKinematics=1
            )[0],
        )
        pybullet.addUserDebugPoints(
            pointPositions=target_points,
            pointColorsRGB=[[1, 0, 0] for _ in range(len(target_points))],
            pointSize=2,
            lifeTime=0.1,
        )
        angles = simulation.calc_inverse_kinematics(
            target_positions=target_points,
            target_orientations=demonstrate_file["ee_ori"][i].tolist(),
        )
        # angles = simulation.ikpy_calculate_inverse_kinematics(
        #     target_positions=target_points,
        #     target_orientations=demonstrate_file["ee_ori"][i].tolist()
        # )
        simulation.step_simulation(angles)
        # simulation.step_simulation([0 for i in range(12)])



    # eef_pose = numpy.identity(4)
    # X = numpy.array([-1.0, 0.0, 0.0])
    # Y = numpy.array([0.0, 1.0, 0.0])
    # Z = numpy.array([0.0, 0.0, -1.0])
    # top_down_orientation = numpy.column_stack([X, Y, Z])
    # translation = numpy.array([0, 0.3, 0.5])
    #
    # eef_pose[:3, :3] = top_down_orientation
    # eef_pose[:3, 3] = translation
    # solutions = ur5e.inverse_kinematics(eef_pose)
    #
    # chain_left = Chain.from_urdf_file(urdf_file="ur_description/ur5_robot_hand.urdf",
    #                                   base_elements=["left_base_link"],
    #                                   active_links_mask=[False if i in [0, 7] else True for i in range(8)])
    # print(chain_left.forward_kinematics([0] + solutions[0].tolist()[0] + [0]))
    # print(ur5e.forward_kinematics(*tuple(solutions[0].tolist()[0])))
    # print(numpy.matrix(numpy.array(ur5e.forward_kinematics(*tuple(solutions[0].tolist()[0]))))@
    #       numpy.matrix(numpy.array(chain_left.forward_kinematics([0] + solutions[0].tolist()[0] + [0]))).I)
    #
    # print('<<<<' * 10)
    #
    # print(ur5e.forward_kinematics(0, 0, 0, 0, 0,0))
    # print(chain_left.forward_kinematics([0 for _ in range(8)]))
    # a=chain_left.forward_kinematics([0 for _ in range(8)])
    # print(numpy.matrix(numpy.array(ur5e.forward_kinematics(0, 0, 0, 0, 0,0))) @
    #       numpy.matrix(numpy.array(a)).I)