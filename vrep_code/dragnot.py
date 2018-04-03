import vrep
import time
import math
import numpy as np
import scipy.linalg as la

################################################################################
# Description of Robot
################################################################################
Base = np.array([0.3250, 0.1250, 0.8053])

# Pose of end effector at the beginning
M = np.array([  [0,0,-1, 0.1049 - Base[0]],
                [0,1,0, 0.1326 - Base[1]],
                [1,0,0, 1.4562 - Base[2]],
                [0,0,0,1] ])

# Screw axes and positions
a = np.array([  [0,0,1],
                [-1,0,0],
                [-1,0,0],
                [-1,0,0],
                [0,0,1],
                [-1,0,0] ])
q = np.array([  [0.3250 - Base[0], 0, 0.9097 - Base[2]],
                [0.2133 - Base[0], 0, 0.9141 - Base[2]],
                [0.2133 - Base[0], 0, 1.11578 - Base[2]],
                [0.2133 - Base[0], 0, 1.3710 - Base[2]],
                [0.2127 - Base[0], 0, 1.4553 - Base[2]],
                [0.2133 - Base[0], 0, 1.4562 - Base[2]] ])

################################################################################
# Helper functions
################################################################################

# Get the current value of the joint variable
def get_joint_val(joint_handle, joint_num, event):
    result, theta = vrep.simxGetJointPosition(clientID, joint_handle, vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get first joint variable')
    #print(event, 'value of joint variable',joint_num, ': theta', joint_num,' = {:f}'.format(theta))
    return theta

# Turn a joint by a given angle
def turn_joint(joint_handle, angle, joint_num, sleep_time):
    time.sleep(sleep_time)
    theta = get_joint_val(joint_handle, joint_num, 'initial')
    vrep.simxSetJointPosition(clientID, joint_handle, theta + angle, vrep.simx_opmode_oneshot)
    time.sleep(sleep_time)

# Sets a joint angle
def set_joint_angle(joint_handle, angle, joint_num, sleep_time):
    time.sleep(sleep_time)
    vrep.simxSetJointPosition(clientID, joint_handle, angle, vrep.simx_opmode_oneshot)
    time.sleep(sleep_time)


# Get "handle" to the given joint of robot
def get_handle(joint_name):
    result, joint_handle = vrep.simxGetObjectHandle(clientID, joint_name, vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get object handle for first joint')
    return joint_handle

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

def open_gripper(clientID):
    #print('Opening gripper...\n')
    vrep.simxSetIntegerSignal(clientID, 'BaxterGripper_close', 0, vrep.simx_opmode_blocking)

def close_gripper(clientID):
    #print('Closing gripper...\n')
    vrep.simxSetIntegerSignal(clientID, 'BaxterGripper_close', 1, vrep.simx_opmode_blocking)

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
def get_current_jacobian(theta):
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

def get_current_pose(theta):
    e1 = la.expm(bracket_screw(S[0]) * theta[0])
    e2 = la.expm(bracket_screw(S[1]) * theta[1])
    e3 = la.expm(bracket_screw(S[2]) * theta[2])
    e4 = la.expm(bracket_screw(S[3]) * theta[3])
    e5 = la.expm(bracket_screw(S[4]) * theta[4])
    e6 = la.expm(bracket_screw(S[5]) * theta[5])
    return  e1.dot(e2).dot(e3).dot(e4).dot(e5).dot(e6).dot(M)

# Sets the position and orientation of an object to the base
def set_object_pose(clientID, pose, obj_name):
    predicted_position = pose[0:3, 3]
    predicted_orient = rot_matrix_to_euler_angles(pose[0:3, 0:3])
    vrep.simxSetObjectPosition(clientID, get_handle(obj_name), get_handle('UR3_link1_visible'), predicted_position, vrep.simx_opmode_oneshot)
    vrep.simxSetObjectOrientation(clientID, get_handle(obj_name), get_handle('UR3_link1_visible'), predicted_orient, vrep.simx_opmode_oneshot)

# Gets the position and orientation of an object relative to the base
def get_object_pose(clientID, obj_name):
    result, position = vrep.simxGetObjectPosition(clientID, get_handle(obj_name), get_handle('UR3_link1_visible'), vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get object position')
    result, euler_angles = vrep.simxGetObjectOrientation(clientID, get_handle(obj_name), get_handle('UR3_link1_visible'), vrep.simx_opmode_blocking)
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

# Returns the distance between two 3D points
def distance_between_points(p_1, p_2):
    return math.sqrt((p_1[0] - p_2[0])**2 + (p_1[1] - p_2[1])**2 + (p_1[2] - p_2[2])**2)

# Given the centers and radii of two spheres, return True if they are colliding
def are_spheres_colliding(p_1, p_2, r_1, r_2):
    return distance_between_points(p_1, p_2) <= (r_1 + r_2)

# Get position of sphere N which is initialially at position M
#  N is a integer either 0, 1, 2, 3, 4, or 5
#  M is a list [x, y, z, 1]
def get_sphere_position(theta, S, M, N):
    e = []

    if N == 0 or N == 1:
        return M

    for i in range(1, N):
        e.append(la.expm(bracket_screw(S[i - 1]) * theta[i - 1]))

    e = np.array(e)
    ret = np.eye(4)
    for i in range(1, N):
        ret = ret.dot(e[i - 1])
    return ret.dot(M)


# HW 5.1.3
# Returns an array sphere_centers where sphere_centers[i] is the center of sphere i
def place_spheres(S, theta, Z):

    S = S.transpose()

    # Spheres 0, 1, ..., 7
    sphere_centers = []
    for i in range(8):
        x = get_sphere_position(theta, S, Z[i], i)
        sphere_centers.append(list(x[:3]))

    sphere_centers = np.array(sphere_centers)
    return sphere_centers

# HW 5.1.4
def is_there_collision(S, theta, Z, r):
    spheres = place_spheres(S, theta, Z)
    #print(spheres)

    collision = False
    for i in range(8):
        for j in range(8):
            if i != j and i != j + 1 and i != j - 1 and i != j + 2 and i != j - 2 and are_spheres_colliding(spheres[i], spheres[j], r, r):
                collision = True
                print(i, j)
    return collision
################################################################################
# Demos
################################################################################

# Checkpoint 2 demo
def forward_kinematics_demo(clientID, joint_handles, S, pose):

    # User inputted thetas (in degrees)
    print("FORWARD KINEMATICS")
    print("Input the values of the six joint angles:")
    print("(e.g. 90 0 0 0 0 0)")
    theta = np.array([math.radians(int(x)) for x in input().split()])
    helper_forward_kinematics(clientID, joint_handles, S, pose, theta)
    print('Initial pose of end effector:\n', pose)

def helper_forward_kinematics(clientID, joint_handles, S, pose, theta):
    for i in range(5, -1, -1):
        #print('i=', i, 'theta=', theta[i])
        bracket_S = bracket_screw(S[i])
        pose = np.dot(la.expm(bracket_S * theta[i]), pose)

    print('Predicted final pose of end effector:\n', pose)

    set_object_pose(clientID, pose, 'Frame_1')

    # Turn all joints
    for i in range(6):
        turn_joint(joint_handles[i], theta[i], i+1, 1)

    # Close gripper
    close_gripper(clientID)

# Returns theta that would produce the given goal pose
def do_inverse_kinematics(S, M, goal_T1in0):
    # Parameters
    #print('Goal pose:', goal_T1in0)
    epsilon = 0.1 # Want error to be this small
    k = 1 # Parameter in theta = theta + thetadot * k
    N = 100 # Maximum number of loops
    mu = 0.1 # Parameter in thetadot equation

    # Initial guess for thetas
    theta = np.array([0,0,0,0,0,0])

    i = 0
    #print('Finding theta that would produce the wanted pose...')
    while True:
        i += 1
        ##print()
        ##print(i)

        # Get current pose that results from theta (forward kinematics)
        current_T1in0 = get_current_pose(theta)
        ##print('current pose:', current_T1in0)

        # Get the twist that would produce the target pose if undergone for 1 second
        bracket_twist = la.logm(goal_T1in0.dot( la.inv(current_T1in0)  ))
        twist = inverse_bracket_screw(bracket_twist)
        ##print('twist:', twist)

        # Compute space Jacobian J of theta
        J = get_current_jacobian(theta)
        ##print('J:', J)

        # Compute thetadot with the regularized least-squares solution
        Jt = np.transpose(J)
        thetadot = la.inv(Jt.dot(J) + mu * np.eye(6)).dot(Jt).dot(twist)
        thetadot = np.squeeze(thetadot)
        ##print('thetadot:', thetadot)

        # Compute new theta = theta + thetadot * 1
        theta = np.add(theta, thetadot * k)
        ##print('newtheta', theta)

        # Determine error
        error = la.norm(twist)
        #print(i, error)

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
        time.sleep(2)
        input("Place block at desired pose and press enter...")

        goal_pose = get_object_pose(clientID, 'Frame_1')

        theta = do_inverse_kinematics(S, M, goal_pose)

        # Turn all joints
        for i in range(6):
            set_joint_angle(joint_handles[i], theta[i], i+1, 0.1)

        # Close gripper
        time.sleep(1)
        close_gripper(clientID)
        time.sleep(1)

    print('END OF INVERSE KINEMATICS DEMO')

# Checkpoint 4 demo
def collision_detection(clientID, joint_handles, S, M):
    print("Collision Detection Demo")
    """
    theta = np.array([[0.14681086, 1.98406767, -0.00064404, 0.54813603, 0.95100498,  -1.33427162, 2.60669557, -1.34891807, 1.77823405, -3.09672843,  -0.02926685, 0.06538319, 1.02944042, 1.78123137, -0.57506472,  -2.48259367, -1.39782694, -1.50441594, -2.34545871], [-2.03078957, -0.99122938, -2.06912857, 2.24969724, -3.05278946,  -0.23510077, -1.02996274, -0.67255515, 0.91400064, 2.69267613,  -2.65655371, 2.97569681, -0.02769013, -0.54810102, 2.20572846,  0.16510457, -2.11613474, 0.62390033, -0.56603554], [0.26982448, 2.30632943, 0.65386007, 1.26331468, -0.99607886,  -0.64818991, -1.00560357, -2.61824931, -0.48405106, 2.07967408,  2.91530733, 3.13671940, 1.93572668, -2.94187968, 0.40684878, 0.86576616,  -2.97548753, -2.85172433, -0.89498416], [-1.92279536, -1.95570646, 1.76049020, 0.92634822, -1.25107838,  2.98196387, -1.79457764, 0.92731661, -0.44421990, -1.90669041,  0.95975764, 1.31289794, -2.00683732, -0.66202909, 0.90891259,  1.68811020, 1.21994831, -1.10110543, -1.71950787], [2.22800557, -1.13467547, 1.71732081, -2.29579612, 0.16752419,  0.57231844, -0.66888501, 0.74781408, 3.11460771, -0.70437482,  0.80792863, -3.06603208, -2.96054108, -1.61436450, -0.55830242,  2.52147902, -2.82611088, 0.82129443, -0.23845192], [-1.65343213, -2.68388272, 2.50329400, 0.87124522, 0.95221881,  2.63061332, 1.40595757, -2.45322703, -0.60805465, 1.71865399,  2.69873775, 0.02218775, 3.09683822, 2.14695020, -2.83026278,  -1.54978798, -1.26556948, 3.13080416, 1.93400129]])
    """
    theta = np.array([[0, 0, 0.1, 0, -0.58787870,  -0.44194958, 1.96679567, -0.50628958, -0.73656183, 3.07307548,  -0.75577025, 2.61955754, -2.98117356, -2.02637222, 2.28347010],
                      [0, 0, 0, 0, 2.03224449,  -1.60210153, 1.32916788, 0.04697692, 0.93594889, 1.44425167, 0.64014588,  1.90738626, 0.82377931, 1.68504340, -1.27063583],
                      [3.14, 1.5, 0, 0, -1.53616482,  1.39426828, -1.31211022, 0.72504312, -2.66573670, 1.60842364,  -0.70850489, 1.92804871, 1.66946648, 1.89600887, -3.05891633],
                      [0, 0, 0, 0, 1.92858218,  -2.39406817, -1.59167495, -2.30805406, 1.86735032, -1.28435563,  0.03837304, -0.98940773, -1.75350804, -0.99173644, 0.63209242],
                      [0, 0, 0, 0, -0.80217197,  2.83709552, 0.45808503, 0.19727700, -0.20277843, -1.03783602,  -0.74450504, -1.33345165, 1.38693544, 1.32363642, 2.08453226],
                      [0, 0, 0, 0, 2.94526677,  -2.61141333, -0.70320318, 2.86590535, 1.51860879, 0.94631448,  2.22162506, 1.85177936, 1.44784656, -2.26096654, 1.03104158]])
    r = 0.05

    # Initial positions of the 8 spheres of the UR3
    Z = []
    Z.append([0, 0, 0, 1]) # First sphere always at zero position

    Z.append([0.3250 - Base[0], 0, 0.9097 - Base[2], 1])
    Z.append([0.2133 - Base[0], 0, 0.9141 - Base[2], 1])
    Z.append([0.2133 - Base[0], 0, 1.11578 - Base[2], 1])
    Z.append([0.2133 - Base[0], 0, 1.3710 - Base[2], 1])
    Z.append([0.2127 - Base[0], 0, 1.4553 - Base[2], 1])
    Z.append([0.2133 - Base[0], 0, 1.4562 - Base[2], 1])

    Z.append([0.1049 - Base[0], 0.1326 - Base[1], 1.4562 - Base[2], 1])

    theta = theta.transpose()

    x = theta[3]
    # Turn all joints
    for i in range(6):
        set_joint_angle(joint_handles[i], x[i], i+1, 0.1)

    c = []
    for i in range(15):
        if is_there_collision(S, theta[i], Z, r):
            c += [1]
        else:
            c += [0]

    c = [c]
    print(c)

    return

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
joint_one_handle = get_handle('UR3_joint1')
joint_two_handle = get_handle('UR3_joint2')
joint_three_handle = get_handle('UR3_joint3')
joint_four_handle = get_handle('UR3_joint4')
joint_five_handle = get_handle('UR3_joint5')
joint_six_handle = get_handle('UR3_joint6')
joint_handles = [joint_one_handle, joint_two_handle, joint_three_handle, joint_four_handle, joint_five_handle, joint_six_handle]

# Start simulation
#vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

################################################################################
#===============================Simulation======================================
################################################################################

# Get array of all S's
S = np.zeros((6, 6))
for i in range(6):
    S[i] = get_screw(a[i], q[i])

# Checkpoint 2 demo
#forward_kinematics_demo(clientID, joint_handles, S, M)

# Checkpoint 3 demo
#inverse_kinematics_demo(clientID, joint_handles, S, M)

################################################################################
#theta = np.array([0,10,20,30,40,50])
#helper_forward_kinematics(clientID, joint_handles, S, M, theta)

collision_detection(clientID, joint_handles, S, M)
################################################################################

# Stop simulation
#vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

# Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
vrep.simxGetPingTime(clientID)

# Close the connection to V-REP
vrep.simxFinish(clientID)
