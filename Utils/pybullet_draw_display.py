import pybullet
import numpy
from scipy.spatial.transform import Rotation

global_lifeTime = 0.08


def set_display_lifetime(lifetime: float) -> None:
    global global_lifeTime
    global_lifeTime = lifetime


def disp_human_demonstrate_arm(target: list[list[float]], draw_bias: list[float]) -> None:
    target = (numpy.array(target) + numpy.array(draw_bias)).tolist()
    color = [1, 0, 1]
    pybullet.addUserDebugLine(
        lineFromXYZ=target[0],
        lineToXYZ=target[1],
        lineColorRGB=color,
        lineWidth=4,
        lifeTime=global_lifeTime,
    )
    pybullet.addUserDebugLine(
        lineFromXYZ=target[1],
        lineToXYZ=target[2],
        lineColorRGB=color,
        lineWidth=4,
        lifeTime=global_lifeTime,
    )


def disp_human_demonstrate_bimanual_arm(target: list[list[float]], draw_bias: list[float]) -> None:
    disp_human_demonstrate_arm(target[:3], draw_bias)
    disp_human_demonstrate_arm(target[3:], draw_bias)


def draw_coordinate(origin_position: list[float], orientation_quaternion: list[float]) -> None:
    orientation_matrix = Rotation.from_quat(orientation_quaternion).as_matrix()
    pybullet.addUserDebugLine(lineFromXYZ=origin_position,
                              lineToXYZ=(orientation_matrix[:, 0].squeeze() + numpy.array(origin_position)).tolist(),
                              lineColorRGB=[1, 0, 0], lineWidth=4, lifeTime=global_lifeTime)
    pybullet.addUserDebugLine(lineFromXYZ=origin_position,
                              lineToXYZ=(orientation_matrix[:, 1].squeeze() + numpy.array(origin_position)).tolist(),
                              lineColorRGB=[0, 1, 0], lineWidth=4, lifeTime=global_lifeTime)
    pybullet.addUserDebugLine(lineFromXYZ=origin_position,
                              lineToXYZ=(orientation_matrix[:, 2].squeeze() + numpy.array(origin_position)).tolist(),
                              lineColorRGB=[0, 0, 1], lineWidth=4, lifeTime=global_lifeTime)
