import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces

class PandaEnvSAC(gym.Env):
    def __init__(self, render_mode=p.DIRECT):
        super(PandaEnvSAC, self).__init__()

        # Connect to PyBullet, load the plane, Panda arm, and object (cube for testing
        # but pill will be used in the final environment)
        self.client = p.connect(render_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1 / 240)  # Set a fixed time step (visualization is smoother)

        self.planeId = p.loadURDF("plane.urdf")
        self.pandaId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        self.cubeId = p.loadURDF("cube_small.urdf", basePosition=[0.7, 0, 0.05])
        self.initial_cube_position, _ = p.getBasePositionAndOrientation(self.cubeId)

        # Increase friction for cube and plane
        p.changeDynamics(self.cubeId, -1, lateralFriction=2.0)
        p.changeDynamics(self.planeId, -1, lateralFriction=2.0)

        # Controlled joints
        self.base_joint = 0                     # Base swivel joint
        self.lower_arm_joint = 1                # Shoulder lift joint
        self.elbow_joint = 3                    # Main elbow joint
        self.wrist_two_joint = 5                # Wrist 2 joint
        self.wrist_three_joint = 6              # Wrist 3 joint
        self.gripper_joint_indices = [9, 10]    # Gripper joints
        self.control_joints = [
            self.base_joint,
            self.lower_arm_joint,
            self.elbow_joint,
            self.wrist_two_joint,
            self.wrist_three_joint
        ] + self.gripper_joint_indices          # Total: 7 joints

        # Fixed joints (non-controlled arm joints)
        self.fixed_joint_positions = {
            2: 0.0,   # Upper arm roll
            4: 0.0    # Wrist 1 joint
        }

        self.end_effector_index = 11
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.control_joints),), dtype=np.float32)

        # Define observation space (joint positions, velocities, cube position, gripper position)
        num_obs = len(self.control_joints) * 2 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_obs,),
            dtype=np.float32
        )

        self.max_episode_steps = 500
        self.current_step = 0

        # Holding parameters
        self.lift_threshold = 0.2  # Cube height threshold to consider it lifted
        self.hold_duration = 50    # Number of steps the cube must be held above the threshold
        self.hold_counter = 0      # Counter for holding the cube

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1 / 240)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.planeId = p.loadURDF("plane.urdf")
        self.pandaId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.cubeId = p.loadURDF("cube_small.urdf", basePosition=[0.7, 0, 0.05])

        p.changeDynamics(self.cubeId, -1, lateralFriction=2.0)
        p.changeDynamics(self.planeId, -1, lateralFriction=2.0)

        # Set initial positions for controlled joints
        # Order: [base, lower_arm, elbow, wrist_two, wrist_three, gripper1, gripper2]
        initial_control_positions = [0.0, -0.5, 0.0, 0.0, 0.0, 0.04, 0.04]

        for idx, joint_index in enumerate(self.control_joints):
            p.resetJointState(self.pandaId, joint_index, targetValue=initial_control_positions[idx])

        # Set fixed joint positions and lock them
        for joint_index, joint_pos in self.fixed_joint_positions.items():
            p.resetJointState(self.pandaId, joint_index, targetValue=joint_pos)
            p.setJointMotorControl2(
                self.pandaId,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=joint_pos,
                force=100 # Not sure wht force to use, used 100 as a placeholder
            )

        self.initial_cube_position, _ = p.getBasePositionAndOrientation(self.cubeId)

        self.current_step = 0
        self.hold_counter = 0

        return self._get_observation()

    def _get_observation(self):
        # Get joint states for controlled joints, cube position, and gripper position
        # then combine them into a single observation
        joint_states = p.getJointStates(self.pandaId, self.control_joints)
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)
        joint_velocities = np.array([state[1] for state in joint_states], dtype=np.float32)

        cube_position, _ = p.getBasePositionAndOrientation(self.cubeId)
        cube_position = np.array(cube_position, dtype=np.float32)

        gripper_state = p.getLinkState(self.pandaId, self.end_effector_index)
        gripper_position = np.array(gripper_state[0], dtype=np.float32)

        observation = np.concatenate([joint_positions, joint_velocities, cube_position, gripper_position])

        return observation

    def step(self, action):
        self.current_step += 1

        # Scale action from [-1, 1] to joint limits
        for idx, joint_index in enumerate(self.control_joints):
            joint_info = p.getJointInfo(self.pandaId, joint_index)
            joint_lower_limit = joint_info[8]
            joint_upper_limit = joint_info[9]

            # Map action from [-1,1] to [joint_lower_limit, joint_upper_limit]
            target_position = joint_lower_limit + (action[idx] + 1) * (
                        joint_upper_limit - joint_lower_limit) / 2

            if joint_index in self.gripper_joint_indices:
                max_force = 100
            else:
                max_force = 150

            p.setJointMotorControl2(
                self.pandaId,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=target_position,
                force=max_force
            )

        p.stepSimulation()
        # time.sleep(1 / 240)  # Comment out during training for faster simulation

        observation = self._get_observation()

        # Extract positions from observation
        num_joints = len(self.control_joints)
        cube_position = observation[num_joints * 2:num_joints * 2 + 3]
        gripper_position = observation[-3:]

        distance_to_cube = np.linalg.norm(cube_position - gripper_position)
        contacts = p.getContactPoints(self.pandaId, self.cubeId)
        cube_displacement = np.linalg.norm(cube_position - self.initial_cube_position)

        # Proximity Reward
        max_distance = 1.0
        threshold = 0.1
        max_proximity_reward = 5.0

        # Determine if the cube has been moved by the gripper
        cube_moved_threshold = 0.05 # Movement threshold
        cube_moved_by_gripper = cube_displacement > cube_moved_threshold and len(contacts) > 0

        if not cube_moved_by_gripper:
            # Penalize more aggressively when the gripper is farther from the cube
            proximity_penalty_scaling = 2.0
            if distance_to_cube > threshold:
                # Negative reward scales with distance
                proximity_reward = - (
                            distance_to_cube / max_distance) * max_proximity_reward * proximity_penalty_scaling
            else:
                # Positive reward scales with closeness
                proximity_reward = (1 - (distance_to_cube / threshold)) * max_proximity_reward
        else:
            # Don't penalize proximity when the cube has been moved by the gripper
            proximity_reward = 0.0

        # Cap the negative proximity reward
        max_negative_proximity_reward = -25.0  # Not sure what value to use
        proximity_reward = max(proximity_reward, max_negative_proximity_reward)

        # Reward for moving the cube
        move_reward = 0.0
        if cube_moved_by_gripper:
            move_reward = 5.0

        # Grasp Reward
        # Check if gripper is closed and in contact with the cube
        gripper_closed = np.all([p.getJointState(self.pandaId, idx)[0] < 0.02 for idx in self.gripper_joint_indices])
        grasp_reward = 0.0
        if gripper_closed and len(contacts) > 0:
            grasp_reward += 20.0

        lifting_reward = 0.0
        gripper_height = gripper_position[2]
        if cube_position[2] > self.lift_threshold:
            base_lift_reward = 15.0
            lifting_reward += base_lift_reward + 0.5 * self.hold_counter

            # More reward for significant lifting (at least 45 degrees)
            if gripper_height > 0.45:
                lifting_reward += 10.0

            self.hold_counter += 1
        else:
            self.hold_counter = 0

        # Total Reward Calculation
        reward = proximity_reward + grasp_reward + lifting_reward + move_reward

        # Print the individual reward components (debugging)
        print("Step: {self.current_step}")
        print("Distance to Cube: {distance_to_cube:.4f}")
        print("Cube Displacement: {cube_displacement:.4f}")
        print("Proximity Reward: {proximity_reward:.4f}")
        print("Move Reward: {move_reward}")
        print("Grasp Reward: {grasp_reward}")
        print("Lifting Reward: {lifting_reward} (Gripper Height: {gripper_height:.4f})")
        print("Hold Counter: {self.hold_counter}")
        print("Total Reward: {reward:.4f}")
        print("-" * 50)

        # Completion Check
        done = False
        success = False
        if self.hold_counter >= self.hold_duration:
            done = True
            success = True
            print("Cube successfully held for required duration.")

        if self.current_step >= self.max_episode_steps:
            done = True

        info = {'is_success': success}

        return observation, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()