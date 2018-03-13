import vrep
import time
import math
import numpy as np
import scipy.linalg as la

Base = np.array([0.3250, 0.1250, 0.8053])

# Pose of end effector at the beginning for forward kinematics
M = np.array([  [0,0,-1, 0.1049 - Base[0] - 0.13],
                [0,1,0, 0.1326 - Base[1]],
                [1,0,0, 1.4562 - Base[2]],
                [0,0,0,1] ])

# Screw axes and positions for forward kinematics
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

# Close all open connections (just in case)
vrep.simxFinish(-1)

# Connect to V-REP (raise exception on failure)
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
if clientID == -1:
    raise Exception('Failed connecting to remote API server')

def turnJoint(joint_handle, angle, joint_num, sleep_time):

    # Wait
    time.sleep(sleep_time)

    # Get the current value of the joint variable
    theta = getJointVal(joint_handle, joint_num, 'initial')

    # Set the desired value of the joint variable
    vrep.simxSetJointTargetPosition(clientID, joint_handle, theta + angle, vrep.simx_opmode_oneshot)

    time.sleep(sleep_time)

    theta = getJointVal(joint_handle, joint_num, 'final')

# Get "handle" to the given joint of robot
def getHandle(joint_name):
    result, joint_handle = vrep.simxGetObjectHandle(clientID, joint_name, vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get object handle for first joint')
    return joint_handle

def getJointVal(joint_handle, joint_num, event):
    result, theta = vrep.simxGetJointPosition(clientID, joint_handle, vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get first joint variable')
    #print(event, 'value of joint variable',joint_num, ': theta', joint_num,' = {:f}'.format(theta))
    return theta

def getScrew(axis, point):
    w = axis
    v = -np.cross(w, point, axis = 0)
    return np.concatenate((w, v), axis = 0)

def getSkewS(screw):
    matrix_w = np.array([
        [0, -screw[2], screw[1]],
        [screw[2], 0,-screw[0]],
        [-screw[1],screw[0], 0]
    ])
    v = (screw[3:6])[:, None]
    matrix_top = np.concatenate((matrix_w, v),axis = 1)
    matrix_bottom = np.array([[0, 0, 0, 0]])
    matrix_screw = np.concatenate((matrix_top, matrix_bottom), axis = 0)
    return matrix_screw

# Get "handles" to all joints of robot
joint_one_handle = getHandle('UR3_joint1')
joint_two_handle = getHandle('UR3_joint2')
joint_three_handle = getHandle('UR3_joint3')
joint_four_handle = getHandle('UR3_joint4')
joint_five_handle = getHandle('UR3_joint5')
joint_six_handle = getHandle('UR3_joint6')
joint_handle= [joint_one_handle, joint_two_handle, joint_three_handle, joint_four_handle, joint_five_handle, joint_six_handle]

# Start simulation
vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

# Forward Kinematics ==========================================================

# User inputted thetas (in degrees)
# e.g. "0 0 0 0 0 0", "90 45 -45 45 -45 45", ""
print("FORWARD KINEMATICS")
print("Input the values of the six joint angles:")
print("(e.g. 90 0 0 0 0 0)")
theta = np.array([math.radians(int(x)) for x in input().split()])
#theta = np.array([np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4])

S = np.zeros((6, 6))
for i in range(6):
    S[i] = getScrew(a[i], q[i])

pose = M
print('Initial pose of end effector:\n', pose)
for i in range(5, -1, -1):
    #print('i=', i, 'theta=', theta[i])
    bracket_S = getSkewS(S[i])
    pose = np.dot(la.expm(bracket_S * theta[i]), pose)

print('Predicted final pose of end effector:\n',pose)

predicted_position = pose[0:3, 3]
vrep.simxSetObjectPosition(clientID, getHandle('Frame_1'), getHandle('UR3_link1_visible'), predicted_position, vrep.simx_opmode_oneshot)
#vrep.simxSetObjectOrientation(clientID, getHandle('Frame_1'), getHandle('UR3_link1_visible'), predicted_orient, simx_opmode_oneshot)

#print(vrep.simxGetObjectPosition(clientID, getHandle('Frame_1'), getHandle('UR3_link1_visible'), vrep.simx_opmode_streaming))
#===============================Simulation======================================

vrep.simxSetIntegerSignal(clientID, 'BaxterGripper_close', 1, vrep.simx_opmode_oneshot)

# Turn all joints
for i in range(6):
    turnJoint(joint_handle[i], theta[i], i+1, 1)
    #print(vrep.simxGetObjectPosition(clientID, getHandle('Frame_1'), getHandle('UR3_link1_visible'), vrep.simx_opmode_buffer))


vrep.simxSetIntegerSignal(clientID, 'BaxterGripper_close', 0, vrep.simx_opmode_oneshot)

vrep.simxSetIntegerSignal(clientID, 'BaxterGripper', 50, vrep.simx_opmode_oneshot)

time.sleep(1000)

#===============================================================================

# Stop simulation
vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

# Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
vrep.simxGetPingTime(clientID)

# Close the connection to V-REP
vrep.simxFinish(clientID)
