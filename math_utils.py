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


def unwind_angles(
    center: float, input_angle: float, period=math.pi * 2, step_size=math.pi * 2
) -> float:
    period = math.fabs(period)
    while input_angle - center > period / 2:
        input_angle = input_angle - step_size
    while input_angle - center < -period / 2:
        input_angle = input_angle + step_size
    return input_angle


def unwind_angle_list(
    center_list: list[float],
    input_list: list[float],
    period=math.pi * 2,
    step_size=math.pi * 2,
) -> list[float]:
    period = math.fabs(period)
    if len(center_list) != len(input_list):
        raise Exception("Angle list length error!")
    return [
        unwind_angles(now_angle, target_angle, period=period, step_size=step_size)
        for now_angle, target_angle in zip(center_list, input_list)
    ]


def point_transfer_scale(
    target: list[list[float]], zero_point: list[float], bias: list[float], scale=1.0
) -> list[list[float]]:
    converted_target = (
        numpy.array(target) - numpy.array(zero_point)
    ) * scale + numpy.array(bias)
    return converted_target.tolist()
