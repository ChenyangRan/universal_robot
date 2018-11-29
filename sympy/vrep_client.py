import vrep as vp

class vrep:
    def __init__(self, ip_address, port, wait_connected, doNotReconnectOnceDisconnected, timeOutInMs, commThreadCycleInMs):
        # close any open connections
        vp.simxFinish(-1)

        # open new connection
        self.clientID = vp.simxStart(ip_address, port, wait_connected, doNotReconnectOnceDisconnected, timeOutInMs, commThreadCycleInMs)
        if self.clientID == -1: # if we connected successfully  # noqa C901
            raise Exception("Failed connecting to remote API server :(")

        self.vp = vp

        print('Connected to remote API server')

    def start_simulation(self, op_mode=vp.simx_opmode_blocking):
        # start our simulation in lockstep with our code
        vp.simxStartSimulation(
            self.clientID,
            op_mode)

    def get_obj_handle(self, names, op_mode=vp.simx_opmode_blocking):
        handle = None
        if type(names) == list:
            handle = [vp.simxGetObjectHandle(
                self.clientID,
                _names,
                op_mode)[1] for _names in names]
        else:
            _, handle = vp.simxGetObjectHandle(
                self.clientID,
                names,
                op_mode)
        return handle

    def set_obj_position(self, handle, pos, relative=-1, op_mode=vp.simx_opmode_blocking):
        _ = vp.simxSetObjectPosition(
        self.clientID,
        handle,
        relative,  # set absolute, not relative position
        pos,
        op_mode)
        if _ != 0:
            raise Exception("Error setting the position of object")

    def get_obj_position(self, handle, relative=-1, op_mode=vp.simx_opmode_blocking):
        _ , pos = vp.simxGetObjectPosition(
            self.clientID,
            handle,
            relative,  # retrieve absolute, not relative, position
            op_mode)
        if _ != 0:
            raise Exception("Error getting the position of object")
        return pos

    def get_joint_position(self, handle, op_mode=vp.simx_opmode_blocking):
        _, pos = vp.simxGetJointPosition(
        self.clientID,
        handle,
        op_mode)
        if _ != 0:
            raise Exception("Joint angle")
        return pos

    def get_obj_param(self, handle, param_id, op_mode=vp.simx_opmode_blocking):
        # get the joint velocity
        _, param = vp.simxGetObjectFloatParameter(
            self.clientID,
            handle,
            param_id,  # parameter ID for angular velocity of the joint
            op_mode)
        if _ != 0:
            raise Exception("Error getting parameter %s" % param_id)
        return param

    def set_floating_parameter(self, dt, time_step=vp.sim_floatparam_simulation_time_step, op_mode=vp.simx_opmode_oneshot):
            vp.simxSetFloatingParameter(
            self.clientID,
            time_step,
            dt,  # specify a simulation time step
            op_mode)

    def stop_simulation(self, op_mode=vp.simx_opmode_blocking):
        # stop the simulation
        vp.simxStopSimulation(self.clientID, op_mode)

    def ping(self):
        # Before closing the connection to V-REP,
        # make sure that the last command sent out had time to arrive.
        vp.simxGetPingTime(self.clientID)

    def set_joint_target_vel(self, handle, velocity, op_mode=vp.simx_opmode_blocking):
        _ = vp.simxSetJointTargetVelocity(
                self.clientID,
                handle,
                velocity,  # target velocity
                op_mode)
        if _ != 0:
            raise Exception("Error setting joint target velocity")

    def get_joint_force(self, handle, op_mode=vp.simx_opmode_blocking):
        _, force = vp.simxGetJointForce(
                    self.clientID,
                    handle,
                    op_mode)
        if _ != 0:
            raise Exception("Error getting the joint force")
        return force

    def set_joint_force(self, handle, force, op_mode=vp.simx_opmode_blocking):
        # and now modulate the force
        _ = vp.simxSetJointForce(
            self.clientID,
            handle,
            force,  # force to apply
            op_mode)
        if _ != 0:
            raise Exception("Error setting joint target force")

    def close_connection(self):
        # Now close the connection to V-REP:
        vp.simxFinish(self.clientID)
        print('connection closed...')

    def synchronous_trigger(self):
        # move simulation ahead one time step
        vp.simxSynchronousTrigger(self.clientID)

    def synchronous(self, enable=True):
        vp.simxSynchronous(self.clientID, enable)

