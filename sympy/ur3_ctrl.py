'''
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import sys
import numpy as np
# np.set_printoptions(precision=5)
import vrep_client

import ur3

# create instance of ur3 class which provides all
# the transform and Jacobian information for this arm
robot_config = ur3.robot_config()

vrep = vrep_client.vrep('127.0.0.1', 19997, True, True, 500, 5)

def velocity_P_control(qd, q, K):
    u = K * (qd - q)
    return u

def print_info():
    # --------------------- Setup the simulation
    vrep.synchronous()

    # --------------------- Start the simulation----------------
    dt = .01
    vrep.set_floating_parameter(dt)  # specify a simulation time step

    # start our simulation in lockstep with our code
    vrep.start_simulation()

    link_names = ['UR3_link%i' % (ii+2) for ii in range(6)]
    joint_names = ['UR3_joint%i' % (ii+1) for ii in range(6)]

    c = list(zip(joint_names, link_names))
    ur3_parts =  [elt for sublist in c for elt in sublist]
    handles = vrep.get_obj_handle(ur3_parts)
    joint_handles = vrep.get_obj_handle(joint_names)
    print ur3_parts

    force = [56,56,28,12,12,12]

    for ii, joint_handle in enumerate(handles):
        vrep.vp.simxSetJointForce(vrep.clientID,joint_handle, force[ii], vrep.vp.simx_opmode_blocking)

    for ii in range(len(handles)):
        print ur3_parts[ii], repr(np.array(vrep.get_obj_position(handles[ii], relative=vrep.vp.sim_handle_parent, op_mode=vrep.vp.simx_opmode_blocking)))
        if ii==11:
            break


    
def control_loop():
    # --------------------- Setup the simulation
    vrep.synchronous()

    # get the handles for each joint and set up streaming
    joint_handles = vrep.get_obj_handle(robot_config.joint_names)
    
    # get handle for hand
    hand_handle = vrep.get_obj_handle('hand')

    # define variables to share with nengo
    q = np.zeros(len(joint_handles))
    dq = np.zeros(len(joint_handles))

    # --------------------- Start the simulation----------------
    dt = .01
    vrep.set_floating_parameter(dt)  # specify a simulation time step

    # start our simulation in lockstep with our code
    vrep.start_simulation()

    count = 0

    controller = velocity_P_control
    K = 500
    qd = [100,0,0,0,0,0]
    # qd = np.deg2rad(qd)

    # NOTE: main loop starts here -----------------------------------------
    while count < 10:

        for ii, joint_handle in enumerate(joint_handles):
            # get the joint angles
            q[ii] = vrep.get_joint_position(joint_handle)
            # get the joint velocities
            param_id = 2012 # parameter ID for angular velocity of the joint
            dq[ii] = vrep.get_obj_param(joint_handle, param_id)

        # calculate position of the end-effector
        # derived in the ur3 calc_TnJ class
        xyz = robot_config.Tx('EE', q)

        # Update position of hand sphere
        vrep.set_obj_position(hand_handle, xyz)

        # print np.round(xyz*100, 2)

        u = controller(qd, q, K)
        print np.array(u)

        for ii, joint_handle in enumerate(joint_handles):
            # get the joint angles
            vrep.set_joint_target_vel(joint_handle, u[ii]) 

        # move simulation ahead one time step
        vrep.synchronous_trigger()
        count += dt
    else:
        raise Exception('Failed connecting to remote API server')

def force_control():
    # --------------------- Setup the simulation
    vrep.synchronous()

    # joint target velocities discussed below
    joint_target_velocities = np.ones(robot_config.num_joints) * 10000.0

    # get the handles for each joint and set up streaming
    joint_handles = vrep.get_obj_handle(robot_config.joint_names)
    
    # get handle for target and set up streaming
    target_handle = vrep.get_obj_handle('target')
    # get handle for hand
    hand_handle = vrep.get_obj_handle('hand')

    # define variables to share with nengo
    q = np.zeros(len(joint_handles))
    dq = np.zeros(len(joint_handles))

    # define a set of targets
    center = np.array([0, 0, 0.6])
    dist = .2
    num_targets = 10
    target_positions = np.array([
        [dist*np.cos(theta)*np.sin(theta),
            dist*np.cos(theta),
            dist*np.sin(theta)] +
        center for theta in np.linspace(0, np.pi*2, num_targets)])
    

    # --------------------- Start the simulation----------------

    # specify a simulation time 
    dt = .01
    vrep.set_floating_parameter(dt)

    # start our simulation in lockstep with our code
    vrep.start_simulation()

    count = 0
    target_index = 0
    change_target_time = dt*1000
    vmax = 0.5
    kp = 200.0
    kv = np.sqrt(kp)

    track_hand = []
    track_target = []

    # NOTE: main loop starts here -----------------------------------------
    while count < 1000:

        # every so often move the target
        if (count % change_target_time) < dt:
            vrep.set_obj_position(target_handle, target_positions[target_index])
            target_index += 1

        # get the (x,y,z) position of the target
        target_xyz = vrep.get_obj_position(target_handle)
        track_target.append(np.copy(target_xyz))  # store for plotting
        target_xyz = np.asarray(target_xyz)

        for ii, joint_handle in enumerate(joint_handles):
            # get the joint angles
            q[ii] = vrep.get_joint_position(joint_handle)
            
            # get the joint velocity
            param_id = 2012 # parameter ID for angular velocity of the joint
            dq[ii] = vrep.get_obj_param(joint_handle, param_id)

        # calculate position of the end-effector
        # derived in the ur3 calc_TnJ class
        xyz = robot_config.Tx('EE', q)

        # calculate the Jacobian for the end effector
        JEE = robot_config.J('EE', q)

        # calculate the inertia matrix in joint space
        Mq = robot_config.Mq(q)

        # calculate the effect of gravity in joint space
        Mq_g = robot_config.Mq_g(q)

        # convert the mass compensation into end effector space
        Mx_inv = np.dot(JEE, np.dot(np.linalg.inv(Mq), JEE.T))
        svd_u, svd_s, svd_v = np.linalg.svd(Mx_inv)
        # cut off any singular values that could cause control problems
        singularity_thresh = .00025
        for i in range(len(svd_s)):
            svd_s[i] = 0 if svd_s[i] < singularity_thresh else \
                1./float(svd_s[i])
        # numpy returns U,S,V.T, so have to transpose both here
        Mx = np.dot(svd_v.T, np.dot(np.diag(svd_s), svd_u.T))

        # calculate desired force in (x,y,z) space
        dx = np.dot(JEE, dq)

        # implement velocity limiting
        lamb = kp / kv
        x_tilde = xyz - target_xyz
        sat = vmax / (lamb * np.abs(x_tilde))
        scale = np.ones(3)
        if np.any(sat < 1):
            index = np.argmin(sat)
            unclipped = kp * x_tilde[index]
            clipped = kv * vmax * np.sign(x_tilde[index])
            scale = np.ones(3) * clipped / unclipped
            scale[index] = 1
        
        u_xyz = -kv * (dx - np.clip(sat / scale, 0, 1) *
                            -lamb * scale * x_tilde)
        u_xyz = np.dot(Mx, u_xyz)

        # transform into joint space, add vel and gravity compensation
        u = np.dot(JEE.T, u_xyz) - Mq_g

        # calculate the null space filter
        Jdyn_inv = np.dot(Mx, np.dot(JEE, np.linalg.inv(Mq)))
        null_filter = (np.eye(robot_config.num_joints) -
                        np.dot(JEE.T, Jdyn_inv))
        # calculate our secondary control signal
        q_des = np.zeros(robot_config.num_joints)
        dq_des = np.zeros(robot_config.num_joints)
        # calculated desired joint angle acceleration using rest angles
        for ii in range(1, robot_config.num_joints):
            if robot_config.rest_angles[ii] is not None:
                q_des[ii] = (
                    ((robot_config.rest_angles[ii] - q[ii]) + np.pi) %
                    (np.pi*2) - np.pi)
                dq_des[ii] = dq[ii]
        # only compensate for velocity for joints with a control signal
        nkp = kp * .1
        nkv = np.sqrt(nkp)
        u_null = np.dot(Mq, (nkp * q_des - nkv * dq_des))

        u += np.dot(null_filter, u_null)

        # get the (x,y,z) position of the center of the obstacle
        v = np.asarray([1,1,1])

        # multiply by -1 because torque is opposite of expected
        u *= -1
        u = [0,0,0,0,10,0]
        print('u: ', u)

        for ii, joint_handle in enumerate(joint_handles):
            # the way we're going to do force control is by setting
            # the target velocities of each joint super high and then
            # controlling the max torque allowed (yeah, i know)

            # get the current joint torque
            torque = vrep.get_joint_force(joint_handle)

            # if force has changed signs,
            # we need to change the target velocity sign
            if np.sign(torque) * np.sign(u[ii]) <= 0:
                joint_target_velocities[ii] = \
                    joint_target_velocities[ii] * -1
                vrep.set_joint_target_vel(
                    joint_handle,
                    joint_target_velocities[ii])  # target velocity

            # and now modulate the force
            vrep.set_joint_force(
                joint_handle,
                abs(u[ii]))  # force to apply

        # Update position of hand sphere
        vrep.set_obj_position(
            hand_handle,
            xyz)
        track_hand.append(np.copy(xyz))  # and store for plotting

        # move simulation ahead one time step
        vrep.synchronous_trigger()
        count += dt

def forward_kinematics():
    # --------------------- Setup the simulation
    vrep.synchronous()

    # get the handles for each joint and set up streaming
    joint_handles = vrep.get_obj_handle(robot_config.joint_names)
    
    # get handle for hand
    hand_handle = vrep.get_obj_handle('hand')
    q4_handle = vrep.get_obj_handle('q4')
    q5_handle = vrep.get_obj_handle('q5')
    q6_handle = vrep.get_obj_handle('q6')

    # define variables to share with nengo
    q = np.zeros(len(joint_handles))

    # --------------------- Start the simulation----------------
    dt = .01
    vrep.set_floating_parameter(dt)  # specify a simulation time step

    # start our simulation in lockstep with our code
    vrep.start_simulation()

    count = 0
    x = 0

    # NOTE: main loop starts here -----------------------------------------
    while count < 10:

        for ii, joint_handle in enumerate(joint_handles):
            # get the joint angles
            q[ii] = vrep.get_joint_position(joint_handle)

        # q[3] = x
        # x += np.deg2rad(5)
        # if x > 2*np.pi:
        #     break
        # calculate position of the end-effector
        # derived in the ur3 calc_TnJ class
        xyz = robot_config.Tx('EE', q)
        pq4 = robot_config.Tx('joint3', q)
        pq5 = robot_config.Tx('joint4', q)
        pq6 = robot_config.Tx('joint5', q)

        # Update position of hand sphere
        vrep.set_obj_position(hand_handle, xyz)
        # vrep.set_obj_position(q4_handle, pq4)
        # vrep.set_obj_position(q5_handle, pq5)
        # vrep.set_obj_position(q6_handle, pq6)

        # print np.round(xyz*100, 2)
        print np.around(pq4*100, 2), np.around(np.rad2deg(x), 1)

        # move simulation ahead one time step
        vrep.synchronous_trigger()
        count += dt
    else:
        raise Exception('Failed connecting to remote API server')

def plot_traj(track_hand, track_target):
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    track_hand = np.array(track_hand)
    track_target = np.array(track_target)
    # track_obstacle = np.array(track_obstacle)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    print "CAM", track_hand
    # plot start point of hand
    ax.plot([track_hand[0, 0]],
            [track_hand[0, 1]],
            [track_hand[0, 2]],
            'bx', mew=10)
    # plot trajectory of hand
    ax.plot(track_hand[:, 0],
            track_hand[:, 1],
            track_hand[:, 2])
    # plot trajectory of target
    ax.plot(track_target[:, 0],
            track_target[:, 1],
            track_target[:, 2],
            'rx', mew=10)
    # # plot trajectory of obstacle
    # ax.plot(track_obstacle[:, 0],
    #         track_obstacle[:, 1],
    #         track_obstacle[:, 2],
    #         'yx', mew=10)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-.5, .5])
    ax.set_zlim([0, 1])
    ax.legend()

    plt.show()

# force_control()
try:
    # forward_kinematics()
    # force_control()
    # control_loop()
    print_info()
finally:
    # stop the simulation
    vrep.stop_simulation()

    # Before closing the connection to V-REP,
    # make sure that the last command sent out had time to arrive.
    vrep.ping()

    # Now close the connection to V-REP:
    vrep.close_connection()

