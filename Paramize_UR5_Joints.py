import pybullet
from UR5_Inverse_Kinematics import UR5_Inverse_Kinematics_Simulation
import math

if __name__ == "__main__":
    simulation = UR5_Inverse_Kinematics_Simulation(
        urdf_file="./ur_description/ur5_robot_hand.urdf", show_gui=True
    )
    joint_angles_parameter_id = [
        pybullet.addUserDebugParameter(
            paramName=f"Joint angle for joint {joint}",
            rangeMin=-2 * math.pi,
            rangeMax=2 * math.pi,
            startValue=0,
        )
        for joint in simulation.available_joints_indices
    ]
    try:
        while True:
            simulation.step_simulation(
                [
                    pybullet.readUserDebugParameter(param_id)
                    for param_id in joint_angles_parameter_id
                ]
            )
    except KeyboardInterrupt:
        pybullet.disconnect(simulation.client)
