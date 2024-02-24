import numpy
import math
from scipy.spatial.transform import Rotation


def vector_dot_loss(a: list[float], b: list[float]) -> float:
    dot = numpy.dot(numpy.array(a), numpy.array(b))
    norm = numpy.linalg.norm(numpy.array(a)) * numpy.linalg.norm(numpy.array(b))
    return 1 - dot / norm


def pack_homogeneous_transfer_matrix(
    translate: list[float], rotation: list[float]
) -> list[list[float]]:
    return numpy.vstack(
        (
            numpy.hstack(
                (
                    Rotation.from_quat(rotation).as_matrix(),
                    numpy.array(translate).reshape(-1, 1),
                )
            ),
            numpy.array([0, 0, 0, 1]),
        )
    ).tolist()


def unpack_homogeneous_transfer_matrix(
    homogeneous_transfer_matrix: list[list[float]],
) -> tuple[list[float], list[list[float]]]:
    return (
        numpy.array(homogeneous_transfer_matrix)[:3, 3].tolist(),
        numpy.array(homogeneous_transfer_matrix)[:3, :3].tolist(),
    )


def unwind_angles(now_angle: float, target_angle: float, period=math.pi * 2) -> float:
    period = math.fabs(period)
    while target_angle - now_angle > period / 2:
        target_angle = target_angle - period
    while target_angle - now_angle < -period / 2:
        target_angle = target_angle + period
    return target_angle


def point_transfer_scale(
    target: list[list[float]], zero_point: list[float], bias: list[float], scale=1.0
) -> list[list[float]]:
    converted_target = (
        numpy.array(target) - numpy.array(zero_point)
    ) * scale + numpy.array(bias)
    return converted_target.tolist()
