import numpy
import math
from scipy.spatial.transform import Rotation
import copy


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


def cvt_target(
        target_point: list[list[float]],
        base_pos: list[float],
        _man_scale: list,
) -> list[list[float]]:
    if _man_scale is None:
        _man_scale = [1.0, 1.0]
    converted_target = copy.deepcopy(target_point)
    converted_target[0] = base_pos
    converted_target[1] = point_transfer_scale(target=[target_point[1]], zero_point=target_point[0],
                                               bias=base_pos, scale=_man_scale[0])[0]
    converted_target[2] = point_transfer_scale(target=[target_point[2]], zero_point=target_point[1],
                                               bias=converted_target[1], scale=_man_scale[1])[0]
    return converted_target


def cvt_target_bimanual(
        target_point: list[list[float]],
        left_base_pos: list[float],
        right_base_pos: list[float],
        _man_scale: list,
) -> list[list[float]]:
    if _man_scale is None:
        _man_scale = [1.0, 1.0]
    return cvt_target(target_point[:3], left_base_pos, _man_scale) + cvt_target(target_point[3:], right_base_pos,
                                                                                _man_scale)


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
    euler_angle[2][1] = euler_angle[1][1]
    euler_angle[2][2] = math.pi - euler_angle[2][2]
    euler_angle[2] = [unwind_angles(0, ang) for ang in euler_angle[2]]

    return euler_angle


def calculate_orientation_error(
        target_orientation_quaternion: list[list[float]],
        now_orientation_quaternion: list[list[float]],
) -> float:
    error = [
        vector_dot_loss(target_ori, now_ori)
        for target_ori, now_ori in zip(target_orientation_quaternion, now_orientation_quaternion)
    ]
    return numpy.linalg.norm(numpy.array(error)) / math.sqrt(2) / 2
