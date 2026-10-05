import math
from dataclasses import dataclass, field

from pyrr import Quaternion, Vector3
from revolve2.modular_robot import ModularRobot
from revolve2.simulation.scene import (
    MultiBodySystem,
    Pose,
    Scene,
    SimulationState,
    UUIDKey,
)

from ._build_multi_body_systems import BodyToMultiBodySystemConverter
from ._convert_terrain import convert_terrain
from ._modular_robot_simulation_handler import ModularRobotSimulationHandler
from ._terrain import Terrain


def physical_orientation(pose: Pose) -> Quaternion:
    """
    Get a robot's true orientation from the pose the simulator reports.

    The MuJoCo simulator hands back a body's orientation by reading MuJoCo's
    (w, x, y, z) quaternion as pyrr's (x, y, z, w). The reported orientation is
    therefore a permutation of the physical one: for a robot that turns about the
    vertical axis it stays a rotation about x, so "the robot's +x axis" comes out
    as the fixed world +x however the robot is facing. This undoes that, giving
    the rotation the robot physically has (verified against MuJoCo's own body
    frame). Use it whenever a direction on the robot is needed.

    :param pose: A robot pose as reported by the simulation state.
    :returns: The robot's true orientation.
    """
    reported = [float(value) for value in pose.orientation]
    return Quaternion([reported[1], reported[2], reported[3], reported[0]])


@dataclass(frozen=True)
class StopOnRobotObjectDistance:
    """Stop a modular robot scene when a robot gets close enough to an object."""

    robot: ModularRobot
    """The robot that should reach the object."""

    obj: MultiBodySystem
    """The object to reach."""

    distance: float
    """The xy-plane distance threshold in metres."""

    reference_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """
    Point on the robot to measure from, in the robot's own frame.

    The default (0, 0, 0) is the robot's core. Set it to the head's offset to
    measure from the head instead; it is rotated with the robot's orientation.
    """

    def bind(
        self,
        modular_robot_to_multi_body_system_mapping: dict[
            UUIDKey[ModularRobot], MultiBodySystem
        ],
    ) -> "_BoundStopOnRobotObjectDistance":
        """
        Bind this stop condition to the converted simulation objects.

        :param modular_robot_to_multi_body_system_mapping: Mapping from modular robot to multi-body system.
        :returns: A simulation-ready stop condition.
        :raises KeyError: If the robot is not in the scene.
        """
        return _BoundStopOnRobotObjectDistance(
            robot_multi_body_system=modular_robot_to_multi_body_system_mapping[
                UUIDKey(self.robot)
            ],
            obj=self.obj,
            distance=self.distance,
            reference_offset=self.reference_offset,
        )


@dataclass(frozen=True)
class _BoundStopOnRobotObjectDistance:
    """Simulation-ready robot-object distance stop condition."""

    robot_multi_body_system: MultiBodySystem
    obj: MultiBodySystem
    distance: float
    reference_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __call__(self, simulation_state: SimulationState) -> bool:
        """
        Check if the robot is within the configured distance of the object.

        :param simulation_state: The current state of the simulation.
        :returns: True when the threshold has been reached.
        """
        robot_pose = simulation_state.get_multi_body_system_pose(
            self.robot_multi_body_system
        )
        robot_pos = robot_pose.position
        if self.reference_offset != (0.0, 0.0, 0.0):
            robot_pos = robot_pos + physical_orientation(robot_pose) * Vector3(
                self.reference_offset
            )
        obj_pos = simulation_state.get_multi_body_system_pose(self.obj).position
        distance = math.sqrt(
            (robot_pos.x - obj_pos.x) ** 2 + (robot_pos.y - obj_pos.y) ** 2
        )
        return distance <= self.distance


@dataclass
class ModularRobotScene:
    """A scene of modular robots in a terrain."""

    terrain: Terrain
    """The terrain of the scene."""

    _robots: list[tuple[ModularRobot, Pose, bool]] = field(default_factory=list)
    """
    The robots in the scene.
    This is an owning collection; the robots are assigned ids when they are added, equal to their index in this list.
    """
    _interactive_objects: list[MultiBodySystem] = field(default_factory=list)
    """Interactive objects in the scene, that are not robots themselves."""

    _stop_conditions: list[StopOnRobotObjectDistance] = field(default_factory=list)
    """Conditions that can stop the simulation before its maximum time."""

    def add_robot(
        self, robot: ModularRobot, pose: Pose = Pose(), translate_z_aabb: bool = True
    ) -> None:
        """
        Add a robot to the scene.

        :param robot: The robot to add.
        :param pose: The pose of the robot.
        :param translate_z_aabb: Whether the robot should be translated upwards so it's T-pose axis-aligned bounding box is exactly on the ground. I.e. if the robot should be placed exactly on the ground. The pose parameters is still added afterwards.
        """
        # Add the robot to the robots list.
        self._robots.append(
            (
                robot,
                Pose(pose.position.copy(), pose.orientation.copy()),
                translate_z_aabb,
            )
        )

    def add_interactive_object(self, objt: MultiBodySystem) -> None:
        """
        Add an intractable object to the scene.

        :param objt: The object as a multi body system.
        """
        self._interactive_objects.append(objt)

    def add_stop_condition(self, stop_condition: StopOnRobotObjectDistance) -> None:
        """
        Add a condition that can stop this scene before its maximum simulation time.

        :param stop_condition: The stop condition to add.
        """
        self._stop_conditions.append(stop_condition)

    def to_simulation_scene(
        self,
    ) -> tuple[Scene, dict[UUIDKey[ModularRobot], MultiBodySystem]]:
        """
        Convert this to a simulation scene.

        :returns: The created scene.
        """
        handler = ModularRobotSimulationHandler()
        scene = Scene(handler=handler)
        modular_robot_to_multi_body_system_mapping: dict[
            UUIDKey[ModularRobot], MultiBodySystem
        ] = {}

        # Add terrain
        scene.add_multi_body_system(convert_terrain(self.terrain))

        # Add robots
        converter = BodyToMultiBodySystemConverter()
        for robot, pose, translate_z_aabb in self._robots:
            # Convert all bodies to multi body systems and add them to the simulation scene
            (
                multi_body_system,
                body_to_multi_body_system_mapping,
            ) = converter.convert_robot_body(
                body=robot.body, pose=pose, translate_z_aabb=translate_z_aabb
            )
            scene.add_multi_body_system(multi_body_system)
            handler.add_robot(
                robot.brain.make_instance(), body_to_multi_body_system_mapping
            )
            modular_robot_to_multi_body_system_mapping[UUIDKey(robot)] = (
                multi_body_system
            )

        for interactive_object in self._interactive_objects:
            scene.add_multi_body_system(interactive_object)

        for stop_condition in self._stop_conditions:
            handler.add_stop_condition(
                stop_condition.bind(modular_robot_to_multi_body_system_mapping)
            )

        return scene, modular_robot_to_multi_body_system_mapping
