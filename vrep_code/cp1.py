import vrep
import time
import numpy as np

# Close all open connections (just in case)
vrep.simxFinish(-1)

# Connect to V-REP (raise exception on failure)
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
if clientID == -1:
    raise Exception('Failed connecting to remote API server')

def turnJoint(joint_handle, angle):
    # Wait two seconds
    time.sleep(2)

    # Get the current value of the joint variable
    theta = getJointVal(joint_handle)

    # Set the desired value of the joint variable
    vrep.simxSetJointTargetPosition(clientID, joint_handle, theta + angle, vrep.simx_opmode_oneshot)
    return

# Get "handle" to the given joint of robot
def getHandle(joint_name):
    result, joint_handle = vrep.simxGetObjectHandle(clientID, joint_name, vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get object handle for first joint')
    return joint_handle

def getJointVal(joint_handle):
    result, theta = vrep.simxGetJointPosition(clientID, joint_handle, vrep.simx_opmode_blocking)
    if result != vrep.simx_return_ok:
        raise Exception('could not get first joint variable')
    print('current value of first joint variable: theta = {:f}'.format(theta))
    return theta


# Get "handle" to the first joint of robot
joint_one_handle = getHandle('UR3_joint1')

# Start simulation
vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

turnJoint(joint_one_handle, np.pi/2)

turnJoint(joint_one_handle, np.pi/2)

# Wait two seconds
time.sleep(2)

# Get the current value of the first joint variable
theta = getJointVal(joint_one_handle)

# Stop simulation
vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

# Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
vrep.simxGetPingTime(clientID)

# Close the connection to V-REP
vrep.simxFinish(clientID)
