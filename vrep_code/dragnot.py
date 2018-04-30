import vrep
import time
import math
import numpy as np
import scipy.linalg as la

################################################################################
# Description of the Robot
################################################################################

# Pose of end effector at the zero configuration (all thetas zero)
M = np.array([  [0, 0, -1, -0.22008017],
                [0, 1, 0, 0.00761688 ],
                [1, 0, 0, 0.65112448],
                [0, 0, 0, 1]])

# Screw axes and positions
a = np.array([  [0,0,1],
                [-1,0,0],
                [-1,0,0],
                [-1,0,0],
                [0,0,1],
                [-1,0,0]])

q = np.array([  [-2.98023224e-08, -7.45058060e-08, 1.04472935e-01],
                [-1.11666068e-01, -3.12328339e-05, 1.08873367e-01],
                [-1.11666396e-01,  4.46438789e-05, 3.52523446e-01],
                [-1.11665919e-01, -4.91738319e-07, 5.65773487e-01],
                [-1.12349987e-01, -5.21540642e-07, 6.49991035e-01],
                [-1.11665905e-01, -5.66244125e-07, 6.51123405e-01]])

################################################################################
# Helper functions that interface with VREP
################################################################################

# Get the current value of the joint variable
def get_joint_val(clientID, joint_handle):
    result, theta = vrep.simxGetJointPosition(clientID, joint_handle, vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get first joint variable')
    return theta

# Turn a joint by a given angle
def turn_joint(clientID, joint_handle, angle, joint_num, sleep_time):
    time.sleep(sleep_time)
    theta = get_joint_val(clientID, joint_handle)
    vrep.simxSetJointPosition(clientID, joint_handle, theta + angle, vrep.simx_opmode_oneshot)
    time.sleep(sleep_time)

# Sets a joint angle
def set_joint_angle(clientID, joint_handle, angle, joint_num, sleep_time):
    time.sleep(sleep_time)
    vrep.simxSetJointPosition(clientID, joint_handle, angle, vrep.simx_opmode_oneshot)
    time.sleep(sleep_time)

# Returns the current robot configuration
def get_robot_configuration(clientID, joint_handles):
    current_configuration = []
    for j in range(6):
        current_configuration.append(get_joint_val(clientID, joint_handles[j]))
    return current_configuration

# Sets a robot configuration by setting all joint angles to the
#  given configuration
def place_robot_in_configuration(clientID, joint_handles, configuration, delay):
    time.sleep(delay)
    vrep.simxPauseCommunication(clientID, True)
    for j in range(6):
        vrep.simxSetJointPosition(clientID, joint_handles[j], configuration[j], vrep.simx_opmode_oneshot)
    vrep.simxPauseCommunication(clientID, False)
    time.sleep(delay)

# Sets a robot configuration by setting all joint angles to the
#   configurations on a straight-line path to the given configuration
def smooth_place_robot_in_configuration(clientID, joint_handles, theta_start, theta_end):
    N = 20; theta = []; slopes = []
    for i in range(len(theta_start)):
        slopes.append((theta_end[i] - theta_start[i]) / N)
    for i in range(N):
        next_theta = []
        for j in range(len(theta_start)):
            next_theta.append(slopes[j] * i + theta_start[j])
        theta.append(next_theta)
    theta.append(theta_end)
    theta = np.array(theta)

    for i in range(len(theta)):
        the_theta = theta[i]
        vrep.simxPauseCommunication(clientID, True)
        for j in range(6):
            vrep.simxSetJointPosition(clientID, joint_handles[j], the_theta[j], vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(clientID, False)
        time.sleep(0.05)

# Get "handle" to the given joint of robot
def get_handle(clientID, joint_name):
    result, joint_handle = vrep.simxGetObjectHandle(clientID, joint_name, vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get object handle for first joint')
    return joint_handle

# Opens the baxer gripper
def open_gripper(clientID):
    vrep.simxSetIntegerSignal(clientID, 'BaxterGripper_close', 0, vrep.simx_opmode_blocking)

# Closes the baxer gripper
def close_gripper(clientID):
    vrep.simxSetIntegerSignal(clientID, 'BaxterGripper_close', 1, vrep.simx_opmode_oneshot)
    return

# Sets the position and orientation of an object relative to the base
def set_object_pose(clientID, pose, obj_name):
    predicted_position = pose[0:3, 3]
    predicted_orient = rot_matrix_to_euler_angles(pose[0:3, 0:3])
    vrep.simxSetObjectPosition(clientID, get_handle(clientID, obj_name), get_handle(clientID, 'UR3_link1_visible'), predicted_position, vrep.simx_opmode_oneshot)
    vrep.simxSetObjectOrientation(clientID, get_handle(clientID, obj_name), get_handle(clientID, 'UR3_link1_visible'), predicted_orient, vrep.simx_opmode_oneshot)

# Gets the position and orientation of an object relative to the base
def get_object_pose(clientID, obj_name):
    result, position = vrep.simxGetObjectPosition(clientID, get_handle(clientID, obj_name), get_handle(clientID, 'UR3_link1_visible'), vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get object position')
    result, euler_angles = vrep.simxGetObjectOrientation(clientID, get_handle(clientID, obj_name), get_handle(clientID, 'UR3_link1_visible'), vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get object orientation')
    #print(position)
    #print(euler_angles)

    Rx = np.array([ [1, 0, 0],
                    [0, math.cos(euler_angles[0]), -math.sin(euler_angles[0])],
                    [0, math.sin(euler_angles[0]), math.cos(euler_angles[0])] ])

    Ry = np.array([ [math.cos(euler_angles[1]),0,math.sin(euler_angles[1])],
                    [0,1,0],
                    [-math.sin(euler_angles[1]),0,math.cos(euler_angles[1])] ])

    Rz = np.array([ [math.cos(euler_angles[2]),-math.sin(euler_angles[2]),0],
                    [math.sin(euler_angles[2]),math.cos(euler_angles[2]),0],
                    [0,0,1] ])

    R = Rx.dot(Ry).dot(Rz)
    position = (np.array(position))[:, None]
    foo_top = np.concatenate((R, position), axis = 1)
    foo_bottom = np.array([[0,0,0,1]])
    pose = np.concatenate((foo_top, foo_bottom), axis = 0)

    return pose

# Gets the position of an object relative to the base
def get_object_position(clientID, obj_name):
    result, position = vrep.simxGetObjectPosition(clientID, get_handle(clientID, obj_name), get_handle(clientID, 'UR3_link1_visible'), vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get object position')
    #print(position)
    return position

################################################################################
# Helper functions that do not interface with VREP
################################################################################

def get_screw(axis, point):
    w = axis
    v = -np.cross(w, point, axis = 0)
    return np.concatenate((w, v), axis = 0)

def inverse_bracket_screw(x):
    return np.array([-x[1][2], x[0][2], -x[0][1], x[0][3], x[1][3], x[2][3]])[:, None]

def bracket_screw(screw):
    matrix_w = np.array([
        [0, -screw[2], screw[1]],
        [screw[2], 0,-screw[0]],
        [-screw[1],screw[0], 0]
    ])
    v = (screw[3:6])[:, None]
    matrix_top = np.concatenate((matrix_w, v), axis = 1)
    matrix_bottom = np.array([[0, 0, 0, 0]])
    matrix_screw = np.concatenate((matrix_top, matrix_bottom), axis = 0)
    return matrix_screw

def rot_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def adjoint(T):
    R = T[:3,:3]
    p = T[:3, 3]

    b_p = np.array([
        [0,-p[2],p[1]],
        [p[2],0,-p[0]],
        [-p[1],p[0],0]
    ])
    foo_leftbottom = np.dot(b_p, R)
    zeroes33 = np.zeros((3,3))
    foo_top = np.concatenate((R, zeroes33),axis = 1)
    foo_bottom = np.concatenate((foo_leftbottom, R),axis = 1)
    adjoint = np.concatenate((foo_top, foo_bottom),axis = 0)
    return adjoint

def cat(a,b,ax): return np.concatenate((a, b), axis = ax)

# can probably be shortened
def get_current_jacobian(S, theta):
    e1 = la.expm(bracket_screw(S[0]) * theta[0])
    e2 = la.expm(bracket_screw(S[1]) * theta[1])
    e3 = la.expm(bracket_screw(S[2]) * theta[2])
    e4 = la.expm(bracket_screw(S[3]) * theta[3])
    e5 = la.expm(bracket_screw(S[4]) * theta[4])
    e6 = la.expm(bracket_screw(S[5]) * theta[5])
    J1 = (S[0])[:, None]
    J2 = np.dot(adjoint(e1), S[1])[:, None]
    J3 = np.dot(adjoint(e1.dot(e2)), S[2])[:, None]
    J4 = np.dot(adjoint(e1.dot(e2).dot(e3)), S[3])[:, None]
    J5 = np.dot(adjoint(e1.dot(e2).dot(e3).dot(e4)), S[4])[:, None]
    J6 = np.dot(adjoint(e1.dot(e2).dot(e3).dot(e4).dot(e5)), S[5])[:, None]
    Js = cat(J1, J2, 1)
    Js = cat(Js, J3, 1)
    Js = cat(Js, J4, 1)
    Js = cat(Js, J5, 1)
    Js = cat(Js, J6, 1)
    return Js

def get_current_pose(S, theta):
    e1 = la.expm(bracket_screw(S[0]) * theta[0])
    e2 = la.expm(bracket_screw(S[1]) * theta[1])
    e3 = la.expm(bracket_screw(S[2]) * theta[2])
    e4 = la.expm(bracket_screw(S[3]) * theta[3])
    e5 = la.expm(bracket_screw(S[4]) * theta[4])
    e6 = la.expm(bracket_screw(S[5]) * theta[5])
    return  e1.dot(e2).dot(e3).dot(e4).dot(e5).dot(e6).dot(M)

# Returns the distance between two 3D points
def distance_between_points(p_1, p_2):
    return math.sqrt((p_1[0] - p_2[0])**2 + (p_1[1] - p_2[1])**2 + (p_1[2] - p_2[2])**2)

# Given the centers and radii of two spheres, return True if they are colliding
def are_spheres_colliding(p_1, p_2, r_1, r_2):
    return distance_between_points(p_1, p_2) <= (r_1 + r_2)

# Get position of robot sphere N (N >= 0)
# which is initialially at position M (e.g. M = np.array([1, 2, 3]))
def get_sphere_position(theta, S, M, N):
    e = []
    theta = theta.squeeze()
    b = np.append(M, [1])
    Q = -1

    if N == 0 or N == 1:
        return b
    elif N >= 2 and N <= 4:
        Q = 2
    elif N >= 5 and N <= 7:
        Q = 3
    elif N == 8:
        Q = 4
    elif N == 9:
        Q = 5
    elif N >= 10:
        Q = 6

    for i in range(1, Q):
        e.append(la.expm(bracket_screw(S[i - 1]) * theta[i - 1]))

    e = np.array(e)
    a = np.eye(4)
    for i in range(1, Q):
        a = a.dot(e[i - 1])

    return a.dot(b)

# Inputs:
#  - S: as given by prairielearn
#  - theta: as given by prairielearn
#  - P_robot: initial centers of the robot spheres
# Output: array sphere_centers where sphere_centers[i] is the
#  center of robot sphere i
def place_robot_spheres(S, theta, P_robot):
    S = S.transpose()

    sphere_centers = []
    for i in range(len(P_robot)):
        x = get_sphere_position(theta, S, P_robot[i], i)
        sphere_centers.append(list(x[:3]))

    sphere_centers = np.array(sphere_centers)
    return sphere_centers

# Inputs:
#  - S: as given by prairielearn
#  - theta: as given by prairielearn
#  - P_robot: initial centers of the spheres covering the robot
#  - P_obstacle: initial centers of the spheres covering the obstacles
#  - r: radii of the spheres
def is_there_collision(S, theta, P_robot, P_obstacle, R):
    robot_spheres = place_robot_spheres(S, theta, P_robot)
    #rint("robot spheres: ", robot_spheres)
    #print(P_obstacle)
    spheres = np.concatenate((robot_spheres, P_obstacle), axis = 0)
    N1 = len(robot_spheres)
    N = N1 + len(P_obstacle)
    collision = 0
    #print(N1, N)

    #print(spheres)
    for i in range(N1):
        for j in range(N):
            if i != j and i != j + 1 and i != j - 1 and are_spheres_colliding(spheres[i], spheres[j], R[i], R[j]):
                if i < N1 and j < N1:
                    #print('Collision detected between', i, 'and', j)
                    collision = 1
                else:
                    collision = 2


    return collision

# Inputs:
#  - theta_start: a list of N thetas where N = number of joints
#  - theta_end: a list of N thetas where N = number of joints
# Output: returns True if there is a collision in the straight-line path
#   in C-space from theta_start to theta_end
def is_there_collision_in_line(S, theta_start, theta_end, P_robot, P_obstacle, R):
    # Generate list of 20 theta lists
    N = 20
    theta = []
    slopes = []
    for i in range(len(theta_start)):
        slopes.append((theta_end[i] - theta_start[i]) / N)
    for i in range(N):
        next_theta = []
        for j in range(len(theta_start)):
            next_theta.append(slopes[j] * i + theta_start[j])
        theta.append(next_theta)
    num_configurations = len(theta)
    theta = np.array(theta)

    # Check collision
    for i in range(num_configurations):
        if is_there_collision(S, theta[i], P_robot, P_obstacle, R):
            return True

    return False

class Node():
    def __init__(self):
        self.parent = []
        self.data = None

# Returns a random sample point in free space
#   N is the dimension of the C-space
def get_theta_sample(S, P_robot, P_obstacle, R, N):
    sample = Node()
    while True:
        sample_data = np.array([np.random.uniform(-math.pi, math.pi) for _ in range(N)])
        if not is_there_collision(S, sample_data, P_robot, P_obstacle, R):
            sample.data = sample_data.tolist()
            return sample

# Returns distance between two nodes in C-space
#  a and b are nodes with data that are lists
def get_distance(a, b):
    return np.linalg.norm(np.array(a.data) - np.array(b.data))

def get_closest_node_in_tree(the_node, tree):
    closest_distance = math.inf
    closest = None
    for node in tree:
        distance = get_distance(the_node, node)
        if distance < closest_distance:
            closest_distance = distance
            closest = node
    return closest

################################################################################
# Demos
################################################################################

# Checkpoint 2 demo
def forward_kinematics_demo(clientID, joint_handles, S, pose):
    print("FORWARD KINEMATICS")
    print("Input the values of the six joint angles:")
    print("(e.g. 90 0 0 0 0 0)")
    theta = np.array([math.radians(int(x)) for x in input().split()])
    helper_forward_kinematics(clientID, joint_handles, S, pose, theta)
    print('Initial pose of end effector:\n', pose)

def helper_forward_kinematics(clientID, joint_handles, S, pose, theta):
    for i in range(5, -1, -1):
        bracket_S = bracket_screw(S[i])
        pose = np.dot(la.expm(bracket_S * theta[i]), pose)

    print('Predicted final pose of end effector:\n', pose)

    set_object_pose(clientID, pose, 'Frame_1')

    # Turn all joints
    for i in range(6):
        turn_joint(clientID, joint_handles[i], theta[i], i+1, 1)

    close_gripper(clientID)

# Returns theta that would produce the given goal pose
# Inputs: S where S[i] is the ith screw
# M is the initial pose of the end effector
# goal_T1in0 is the goal pose of the end effector
def do_inverse_kinematics(S, M, goal_T1in0):
    # Parameters
    epsilon = 0.01 # Want error to be this small
    k = 1 # Parameter in theta = theta + thetadot * k
    N = 200 # Maximum number of loops
    mu = 0.1 # Parameter in thetadot equation

    # Initial guess for thetas
    theta = np.array([0,0,0,0,0,0])
    i = 0
    while True:
        i += 1

        # Get current pose that results from theta (forward kinematics)
        current_T1in0 = get_current_pose(S, theta)

        # Get the twist that would produce the target pose if undergone for 1 second
        bracket_twist = la.logm(goal_T1in0.dot( la.inv(current_T1in0)  ))
        twist = inverse_bracket_screw(bracket_twist)

        # Compute space Jacobian J of theta
        J = get_current_jacobian(S, theta)

        # Compute thetadot with the regularized least-squares solution
        Jt = np.transpose(J)
        thetadot = la.inv(Jt.dot(J) + mu * np.eye(6)).dot(Jt).dot(twist)
        thetadot = np.squeeze(thetadot)

        # Compute new theta = theta + thetadot * 1
        theta = np.add(theta, thetadot * k)

        # Determine error
        error = la.norm(twist)

        # Stop if error is small enough or if we have looped too many times
        if error < epsilon:
            theta = theta[:, None]
            print('Success! theta = ', theta.tolist())
            break
        if i > N:
            theta = theta[:, None]
            print('Not reachable.', error)
            break

    return theta

# Checkpoint 3 demo
def inverse_kinematics_demo(clientID, joint_handles, S, M):
    print('INVERSE KINEMATICS DEMO')

    while True:
        open_gripper(clientID)
        pos = input("Enter the chessboard position : ")

        goal_pose = get_object_pose(clientID, pos)
        goal_pose[0, 0] = -1;
        goal_pose[2, 2] = -1;
        goal_pose[2, 3] += 0.1;
        print(goal_pose)

        theta = do_inverse_kinematics(S.T, M, goal_pose)

        # Turn all joints
        theta_start = get_robot_configuration(clientID, joint_handles)
        smooth_place_robot_in_configuration(clientID, joint_handles, theta_start, theta)

        # Close gripper
        time.sleep(3)
        close_gripper(clientID)
        time.sleep(1)

    print('END OF INVERSE KINEMATICS DEMO')

# Checkpoint 4 demo
def collision_detection(clientID, joint_handles, S, M, P_robot, P_obstacle, r_robot, r_obstacle):
    x = [0, 0, 0, 0, 0, 0]
    for j in range(6):
        set_joint_angle(clientID, joint_handles[j], x[j], j+1, 0.1)

    print("Collision Detection Demo")

    theta = np.array([[0, 0.15, 0.3, 0.45, 0.60, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5,
    -0.44194958, 1.96679567, -0.50628958, -0.73656183, 3.07307548,  -0.75577025, 2.61955754, -2.98117356, -2.02637222, 2.28347010],
                      [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
    -0.60210153, 0.32916788, 0.04697692, 0.93594889, 0.44425167, 0.64014588,  0.90738626, 0.82377931, 0.68504340, -0.27063583],
                      [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    1.39426828, -1.31211022, 0.72504312, -2.66573670, 1.60842364,  -0.70850489, 1.92804871, 1.66946648, 1.89600887, -3.05891633],
                      [0, 0.15, 0.3, 0.45, 0.60, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5,
    -2.39406817, -1.59167495, -2.30805406, 1.86735032, -1.28435563,  0.03837304, -0.98940773, -1.75350804, -0.99173644, 0.63209242],
                      [0, 0.15, 0.3, 0.45, 0.60, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5,
    2.83709552, 0.45808503, 0.19727700, -0.20277843, -1.03783602,  -0.74450504, -1.33345165, 1.38693544, 1.32363642, 2.08453226],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -2.61141333, -0.70320318, 2.86590535, 1.51860879, 0.94631448,  2.22162506, 1.85177936, 1.44784656, -2.26096654, 1.03104158]])
    r = 0.05

    # Radii of all the spheres
    R = np.concatenate((r_robot, r_obstacle), axis = 1)
    R = R.squeeze()

    theta = theta.transpose()

    for i in range(len(theta)):
        print(i)
        x = theta[i]
        # Turn all joints
        for j in range(6):
            set_joint_angle(clientID, joint_handles[j], x[j], j+1, 0.1)
        print("Turned joints to configuration ", i)

        coll = is_there_collision(S, x, P_robot, P_obstacle, R)
        if coll == 1:
            print("SELF COLLISION!")
        elif coll == 2:
            print("COLLISION WITH OBSTACLE!")
        else:
            print("No collision.")

        x = [0, 0, 0, 0, 0, 0]
        for j in range(6):
            set_joint_angle(clientID, joint_handles[j], x[j], j+1, 0.1)

# Checkpoint 4.5 demo
def checking_collision_in_line(clientID, joint_handles, S, M, P_robot, P_obstacle, r_robot, r_obstacle):
    theta_start = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]])
    theta_goal = np.array([[-0.09, -2.68, 0.55, 0.17, 0.97, 0.80, 1.12],
                           [0, 0.41, 0.12, 2.79, 0.52, 2.27, 1.13],
                           [0.27, 2.49, 0.84, 3.10, -1.42, 2.86, 1.30],
                           [0.11, -1.17, 0.79, 1.61, -0.78, -0.34, -2.68],
                           [0.27, 2.49, 0.84, 3.10, -1.42, 2.86, 1.30],
                           [0.11, -1.17, 0.79, 1.61, -0.78, -0.34, -2.68]])

    theta_start = theta_start.transpose()
    theta_end = theta_goal.transpose()

    # Radii of all the spheres
    R = np.concatenate((r_robot, r_obstacle), axis = 1)
    R = R.squeeze()

    for i in range(6):
        # Put robot in start configuration
        place_robot_in_configuration(clientID, joint_handles, theta_start[i], 0.1)
        print("Placed robot in theta_start configuration")

        if is_there_collision_in_line(S, theta_start[i], theta_goal[i], P_robot, P_obstacle, R):
            print("Collision detected in line from theta_start to theta_goal")
        else:
            print("No collision detected in line from theta_start to theta_goal")
            smooth_place_robot_in_configuration(clientID, joint_handles, theta_start[i], theta_goal[i])
            print("Placed robot in theta_goal configuration")

        input("Press enter to continue...")
        print()

# Checkpoint 5 demo
def motion_planning(clientID, joint_handles, S, M, P_robot, P_obstacle, r_robot, r_obstacle):
    print("Motion Planning Demo")
    theta_start = np.array([[0], [0], [0], [0], [0], [0]])
    theta_goal = np.array([[0.24], [0.01], [-0.32], [0.94], [1.09], [0.93]])

    # Put robot in start configuration
    place_robot_in_configuration(clientID, joint_handles, theta_start, 0.1)

    # Radii of all the spheres
    R = np.concatenate((r_robot, r_obstacle), axis = 1)
    R = R.squeeze()

    #position_ball = np.array([+2.3000e-02 - Base[0], +5.8900e-01 - Base[1], +1.0250e+00 - Base[2]])
    goal_pose = get_object_pose(clientID, 'Target')
    print(goal_pose)
    theta = do_inverse_kinematics(S.T, M, goal_pose)
    print(theta)
    place_robot_in_configuration(clientID, joint_handles, theta, 0.1)
    #smooth_place_robot_in_configuration(clientID, joint_handles, theta_start, theta)

    #smooth_place_robot_in_configuration(clientID, joint_handles, theta, theta_start)


################################################################################
#===============================Initalization===================================
################################################################################

# Close all open connections (just in case)
vrep.simxFinish(-1)

# Connect to V-REP (raise exception on failure)
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
if clientID == -1:
    raise Exception('Failed connecting to remote API server')

# Get "handles" to all joints of robot
joint_one_handle = get_handle(clientID, 'UR3_joint1')
joint_two_handle = get_handle(clientID, 'UR3_joint2')
joint_three_handle = get_handle(clientID, 'UR3_joint3')
joint_four_handle = get_handle(clientID, 'UR3_joint4')
joint_five_handle = get_handle(clientID, 'UR3_joint5')
joint_six_handle = get_handle(clientID, 'UR3_joint6')
joint_handles = [joint_one_handle, joint_two_handle, joint_three_handle, joint_four_handle, joint_five_handle, joint_six_handle]

# Start simulation
vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

################################################################################
#===============================Simulation======================================
################################################################################

# Reset position to all zeros
theta = [0, 0, 0, 0, 0, 0]
theta_start = get_robot_configuration(clientID, joint_handles)
smooth_place_robot_in_configuration(clientID, joint_handles, theta_start, theta)

# Get array of all S's
S = np.zeros((6, 6))
for i in range(6):
    S[i] = get_screw(a[i], q[i])
S = S.transpose()

# Checkpoint 2 demo
#forward_kinematics_demo(clientID, joint_handles, S, M)

# Checkpoint 3 demo


inverse_kinematics_demo(clientID, joint_handles, S, M)

# Initial positions of the spheres of the UR3
P_robot = []
P_robot.append(get_object_position(clientID, 'Dummy_0'))
P_robot.append(get_object_position(clientID, 'Dummy_1'))
P_robot.append(get_object_position(clientID, 'Dummy_2'))
P_robot.append(get_object_position(clientID, 'Dummy_3'))
P_robot.append(get_object_position(clientID, 'Dummy_4'))
P_robot.append(get_object_position(clientID, 'Dummy_5'))
P_robot.append(get_object_position(clientID, 'Dummy_6'))
P_robot.append(get_object_position(clientID, 'Dummy_7'))
P_robot.append(get_object_position(clientID, 'Dummy_8'))
P_robot.append(get_object_position(clientID, 'Dummy_9'))
P_robot.append(get_object_position(clientID, 'Dummy_10'))
P_robot.append(get_object_position(clientID, 'Dummy_11'))

# Inital position of obstacles
P_obstacle = []
P_obstacle.append(get_object_position(clientID, 'Dummy_12'))

# Radii of robot spheres
r_robot = np.array([[0.075, 0.075, 0.075, 0.075, 0.075, 0.075,
                    0.06, 0.06, 0.06, 0.06, 0.06,
                    0.05]])

# Radii of obstacle spheres (one obstacle for now)
r_obstacle = np.array([[0.15]])

# Checkpoint 4 demo
#collision_detection(clientID, joint_handles, S, M, P_robot, P_obstacle, r_robot, r_obstacle)

# Checkpoint 5 demo
#checking_collision_in_line(clientID, joint_handles, S, M, P_robot, P_obstacle, r_robot, r_obstacle)
#motion_planning(clientID, joint_handles, S, M, P_robot, P_obstacle, r_robot, r_obstacle)

# Stop simulation
vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

# Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
vrep.simxGetPingTime(clientID)

# Close the connection to V-REP
vrep.simxFinish(clientID)
