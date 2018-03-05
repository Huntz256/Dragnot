import vrep
import time
import numpy as np

# Close all open connections (just in case)
vrep.simxFinish(-1)

# Connect to V-REP (raise exception on failure)
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
if clientID == -1:
    raise Exception('Failed connecting to remote API server')

def turnJoint(joint_handle, angle, joint_num, sleep_time):
    # Wait two seconds
    time.sleep(sleep_time)

    # Get the current value of the joint variable
    theta = getJointVal(joint_handle, joint_num, 'initial')

    # Set the desired value of the joint variable
    vrep.simxSetJointTargetPosition(clientID, joint_handle, theta + angle, vrep.simx_opmode_oneshot)

    time.sleep(2)

    theta = getJointVal(joint_handle, joint_num, 'final')
    return

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
    print(event, 'value of joint variable',joint_num, ': theta', joint_num,' = {:f}'.format(theta))
    return theta


# Get "handles" to all joints of robot
joint_one_handle = getHandle('UR3_joint1')
joint_two_handle = getHandle('UR3_joint2')
joint_three_handle = getHandle('UR3_joint3')
joint_four_handle = getHandle('UR3_joint4')
joint_five_handle = getHandle('UR3_joint5')
joint_six_handle = getHandle('UR3_joint6')

# Start simulation
vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

#===============================Simulation======================================

vrep.simxSetIntegerSignal(clientID, 'BaxterGripper_close', 1, vrep.simx_opmode_oneshot)

turnJoint(joint_five_handle, np.pi/2, 5, 2)

turnJoint(joint_five_handle, -np.pi/2, 5, 5)

vrep.simxSetIntegerSignal(clientID, 'BaxterGripper_close', 0, vrep.simx_opmode_oneshot)

turnJoint(joint_one_handle, np.pi/2, 1, 1)

turnJoint(joint_two_handle, np.pi/4, 2, 2)

turnJoint(joint_three_handle, -np.pi/4, 3, 2)

turnJoint(joint_four_handle, np.pi/4, 4, 2)

turnJoint(joint_five_handle, np.pi/4, 5, 2)

turnJoint(joint_six_handle, np.pi/2, 6, 2)

vrep.simxSetIntegerSignal(clientID, 'BaxterGripper', 50, vrep.simx_opmode_oneshot)

time.sleep(4)
#===============================================================================

# Stop simulation
vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

# Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
vrep.simxGetPingTime(clientID)

# Close the connection to V-REP
vrep.simxFinish(clientID)
