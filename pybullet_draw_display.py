import pybullet
import numpy
from scipy.spatial.transform import Rotation

global_lifeTime = 0.1


def disp_human_demonstrate(target: list[list[float]], draw_bias: list[float]) -> None:
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
    pybullet.addUserDebugLine(
        lineFromXYZ=target[3],
        lineToXYZ=target[4],
        lineColorRGB=color,
        lineWidth=4,
        lifeTime=global_lifeTime,
    )
    pybullet.addUserDebugLine(
        lineFromXYZ=target[4],
        lineToXYZ=target[5],
        lineColorRGB=color,
        lineWidth=4,
        lifeTime=global_lifeTime,
    )


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
