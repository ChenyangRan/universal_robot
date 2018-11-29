import cloudpickle
import os
import numpy as np
import sympy as sp


class robot_config:
    def __init__(self):
        self.num_joints = 6
        self.num_links = 6
        self.config_folder = 'ur3_config'

        # create function dictionaries
        self._Tx = {}  # for transform calculations
        self._T_inv = {}  # for inverse transform calculations
        self._J = {}  # for Jacobian calculations

        self._CoM = [] # center of mass of each joint
        self._M = []  # placeholder for (x,y,z) inertia matrices
        self._Mq = None  # placeholder for joint space inertia matrix function
        self._Mq_g = None  # placeholder for joint space gravity term function

        # set up our joint angle symbols
        self.q = [sp.Symbol('q%i' % ii) for ii in range(self.num_joints)]
        self.dq = [sp.Symbol('dq%i' % ii) for ii in range(self.num_joints)]
        # set up an (x,y,z) offset
        self.x = [sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('z')]

        self.gravity = sp.Matrix([[0, 0, -9.81, 0, 0, 0]]).T

        self.link_names = ['link%i' % (ii+1) for ii in range(self.num_links)]
        self.q_names = ['joint%i' % (ii+1) for ii in range(self.num_links)]
        self.joint_names = ['UR3_joint%i' % (ii+1)
                            for ii in range(self.num_joints)]

        # for the null space controller, keep arm near these angles
        self.rest_angles = np.array([None,
                                     np.pi/4.0,
                                     -np.pi/2.0,
                                     np.pi/4.0,
                                     np.pi/2.0,
                                     np.pi/2.0])

        # create the inertia matrices for each link of the ur3
        # Diagonal matrix [m m m Ixx Iyy Izz]
        self._M.append(np.diag(3*[2.0] +[0.00167, 0.001762, 0.001238]))  # link1
        self._M.append(np.diag(3*[3.42]+[0.08478, 0.0874, 0.009851]))  # link2
        self._M.append(np.diag(3*[1.26]+[0.0597, 0.06091, 0.0005315]))  # link3
        self._M.append(np.diag(3*[0.80]+[0.007461, 0.007624, 0.004633]))  # link4
        self._M.append(np.diag(3*[0.80]+[0.004669, 0.007785, 0.0076]))  # link5
        self._M.append(np.diag(3*[0.35]+[0.003870, 0.002621, 0.002621]))  # link6 

        self._CoM.append([0.0,  -0.02, 0])
        self._CoM.append([0.13,  0.0,  0.1157])
        self._CoM.append([0.05,  0.0,  0.0238])
        self._CoM.append([0.0,   0.0,  0.01])
        self._CoM.append([0.0,   0.0,  0.01])
        self._CoM.append([0.0,   0.0,  -0.02])
        self._CoM.append([0.0,   0.0,  0.0]) ## end effector

        self._calc_T()

        Jw = sp.Matrix([0,0,1])
        self.J_orientation = [
            self.T01[:3, :3] * Jw,  # joint 0 orientation
            self.T02[:3, :3] * Jw,  # joint 1 orientation
            self.T03[:3, :3] * Jw,  # joint 2 orientation
            self.T04[:3, :3] * Jw,  # joint 3 orientation
            self.T05[:3, :3] * Jw,  # joint 4 orientation
            self.T06[:3, :3] * Jw]  # joint 5 orientation
    
    def Tx(self, name, q, x=[0, 0, 0]):
        """ Calculates the transform for a joint or link

        name string: name of the joint or link, or end-effector
        q list: set of joint angles to pass in to the T function
        x list: the [x,y,z] position of interest in "name"'s reference frame
        """
        # check for function in dictionary
        if self._Tx.get(name, None) is None:
            print('Generating transform function for %s' % name)
            # TODO: link0 and joint0 share a transform, but will
            # both have their own transform calculated with this check
            self._Tx[name] = self._calc_Tx(
                name, x=x)
        parameters = tuple(q) + tuple(x)
        return self._Tx[name](*parameters)[:-1].flatten()

    def T_inv(self, name, q, x=[0, 0, 0]):
        """ Calculates the inverse transform for a joint or link

        q list: set of joint angles to pass in to the T function
        """
        # check for function in dictionary
        if self._T_inv.get(name, None) is None:
            print('Generating inverse transform function for % s' % name)
            self._T_inv[name] = self._calc_T_inv(
                name=name, x=x)
        parameters = tuple(q) + tuple(x)
        return self._T_inv[name](*parameters)


    def J(self, name, q, x=[0, 0, 0]):
        """ Calculates the transform for a joint or link

        name string: name of the joint or link, or end-effector
        q np.array: joint angles
        """
        # check for function in dictionary
        if self._J.get(name, None) is None:
            print('Generating Jacobian function for %s' % name)
            self._J[name] = self._calc_J(
                name, x=x)
        parameters = tuple(q) + tuple(x)
        return np.array(self._J[name](*parameters))

    def Mq(self, q):
        """ Calculates the joint space inertia matrix for the ur5

        q np.array: joint angles
        """
        # check for function in dictionary
        if self._Mq is None:
            print('Generating inertia matrix function')
            self._Mq = self._calc_Mq()
        parameters = tuple(q) + (0, 0, 0)
        return np.array(self._Mq(*parameters))

    def _calc_Mq(self, lambdify=True):
        """ Uses Sympy to generate the inertia matrix in
        joint space for the ur5

        lambdify boolean: if True returns a function to calculate
                          the Jacobian. If False returns the Sympy
                          matrix
        """

        # check to see if we have our inertia matrix saved in file
        if os.path.isfile('%s/Mq' % self.config_folder):
            Mq = cloudpickle.load(open('%s/Mq' % self.config_folder, 'rb'))
        else:
            # get the Jacobians for each link's COM
            J = [self._calc_J('link%s' % (ii+1), self._CoM[ii], lambdify=False)
                 for ii in range(self.num_links)] 
            

            # transform each inertia matrix into joint space
            # sum together the effects of arm segments' inertia on each motor
            Mq = sp.zeros(self.num_joints)
            for ii in range(self.num_links):
                Mq += sp.Matrix(J[ii].T) * sp.Matrix(self._M[ii]) * sp.Matrix(J[ii])
            Mq = sp.Matrix(Mq)

            if lambdify:
                Mq = sp.lambdify(self.q + self.x, Mq)

            # save to file
            cloudpickle.dump(Mq, open('%s/Mq' % self.config_folder, 'wb'))

        if lambdify is False:
            return Mq
        return Mq

    def Mq_g(self, q):
        """ Calculates the force of gravity in joint space for the ur5

        q np.array: joint angles
        """
        # check for function in dictionary
        if self._Mq_g is None:
            print('Generating gravity effects function')
            self._Mq_g = self._calc_Mq_g()
        parameters = tuple(q) + (0, 0, 0)
        return np.array(self._Mq_g(*parameters)).flatten()

    def _calc_Mq_g(self, lambdify=True):
        """ Uses Sympy to generate the force of gravity in
        joint space for the ur5

        lambdify boolean: if True returns a function to calculate
                          the Jacobian. If False returns the Sympy
                          matrix
        """

        # check to see if we have our gravity term saved in file
        if os.path.isfile('%s/Mq_g' % self.config_folder):
            Mq_g = cloudpickle.load(open('%s/Mq_g' % self.config_folder,
                                         'rb'))
        else:
            # get the Jacobians for each link's COM
            J = [self._calc_J('link%s' % (ii+1), self._CoM[ii], lambdify=False)
                 for ii in range(self.num_links)]

            # transform each inertia matrix into joint space and
            # sum together the effects of arm segments' inertia on each motor
            Mq_g = sp.zeros(self.num_joints, 1)
            for ii in range(self.num_joints):
                Mq_g += J[ii].T * self._M[ii] * self.gravity
            Mq_g = sp.simplify(sp.Matrix(Mq_g))
            
            if lambdify:
                Mq_g = sp.lambdify(self.q + self.x, Mq_g)

            # save to file
            cloudpickle.dump(Mq_g, open('%s/Mq_g' % self.config_folder,
                                        'wb'))

        # if lambdify is False:
        #     return Mq_g
        return Mq_g

    def _calc_J(self, name, x, lambdify=True):
        """ Uses Sympy to generate the Jacobian for a joint or link

        name string: name of the joint or link, or end-effector
        lambdify boolean: if True returns a function to calculate
                          the Jacobian. If False returns the Sympy
                          matrix
        """

        # check to see if we have our Jacobian saved in file
        if os.path.isfile('%s/%s.J' % (self.config_folder, name)):
            J = cloudpickle.load(open('%s/%s.J' %
                                 (self.config_folder, name), 'rb'))
        else:
            Tx = self._calc_Tx(name, x=x, lambdify=False)
            J = []

            # calculate derivative of (x,y,z) wrt to eTx = self._calc_Tx(name, x=x, lambdify=lambdify)ach joint
            for ii in range(self.num_joints):
                J.append([])
                J[ii].append(Tx[0].diff(self.q[ii]))  # dx/dq[ii]
                J[ii].append(Tx[1].diff(self.q[ii]))  # dy/dq[ii]
                J[ii].append(Tx[2].diff(self.q[ii]))  # dz/dq[ii]

            end_point = name.strip('link').strip('joint')
            if end_point != "EE": # EE or index 6 shoud not compute rotation
                end_point = min(int(end_point) + 1, self.num_joints)
                # add on the orientation information up to the last joint
                for ii in range(end_point):
                    J[ii] = J[ii] + self.J_orientation[ii].reshape(1,3).tolist()[0]
                # fill in the rest of the joints orientation info with 0
                for ii in range(end_point, self.num_joints):
                    J[ii] = J[ii] + [0, 0, 0]

            # save to file
            cloudpickle.dump(J, open('%s/%s.J' %
                                     (self.config_folder, name), 'wb'))

        J = sp.Matrix(J).T  # correct the orientation of J
        if lambdify is False:
            return J
        return sp.lambdify(self.q + self.x, J)

    def _calc_T(self):
        # segment lengths associated with each joint
        # dh = [0.1045,-0.2437,-0.2133,0.0842,0.0013,0.0664]
        dh = [0.1519,-0.24365,-0.21325,0.11235,0.08535,0.08190]
        eef = 0

        self.T01 =             self._compute_dh_matrix(0.,  sp.pi/2,   dh[0],  self.q[0]-sp.pi/2)
        self.T02 =  self.T01 * self._compute_dh_matrix(dh[1],    0.,      0.,  self.q[1]-sp.pi/2)
        self.T03 =  self.T02 * self._compute_dh_matrix(dh[2],    0.,      0.,  self.q[2])
        self.T04 =  self.T03 * self._compute_dh_matrix(0.,  sp.pi/2,   dh[3],  self.q[3]-sp.pi/2)
        self.T05 =  self.T04 * self._compute_dh_matrix(0., -sp.pi/2,   dh[4],  self.q[4])
        self.T06 =  self.T05 * self._compute_dh_matrix(0.,       0.,   dh[5],  self.q[5])
        self.T0EE = self.T06
        self.T = [self.T01,self.T02,self.T03,self.T04,self.T05,self.T06,self.T0EE]
        self.T = sp.simplify(self.T)


    def _compute_dh_matrix(self, r, alpha, d, theta):
        A = [[sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha),   r*sp.cos(theta)],
             [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha),  -sp.cos(theta)*sp.sin(alpha),  r*sp.sin(theta)],
             [      0,               sp.sin(alpha),                sp.cos(alpha),                 d       ],
             [      0,               0,                  0,                 1       ]]
        return sp.Matrix(A)

    def _get_T(self, name):
        if name == "EE":
            T = self.T[self.link_names.index('link6')]
        else:
            try:
                T = self.T[self.link_names.index(name)] 
            except:
                T = self.T[self.q_names.index(name)] 
        return T

    def _calc_Tx(self, name, x, lambdify=True):
        """ Uses Sympy to transform x from the reference frame of a joint
        or link to the origin (world) coordinates.

        name string: name of the joint or link, or end-effector
        x list: the [x,y,z] position of interest in "name"'s reference frame
        lambdify boolean: if True returns a function to calculate
                          the transform. If False returns the Sympy
                          matrix
        """

        # check to see if we have our transformation saved in file
        if (os.path.isfile('%s/%s.T' % (self.config_folder, name))) and False:
            print "hola"
            Tx = cloudpickle.load(open('%s/%s.T' %
                                       (self.config_folder, name), 'rb'))
        else:
            T = self._get_T(name)
            # transform x into world coordinates
            Tx = T * sp.Matrix(self.x + [1])
            # save to file
            cloudpickle.dump(Tx, open('%s/%s.T' %
                                      (self.config_folder, name), 'wb'))

        if lambdify is False:
            return Tx
        return sp.lambdify(self.q + self.x, Tx)

    def _calc_T_inv(self, name, x, lambdify=True):
        """ Return the inverse transform matrix, which converts from
        world coordinates into the robot's end-effector reference frame

        name string: name of the joint or link, or end-effector
        x list: the [x,y,z] position of interest in "name"'s reference frame
        lambdify boolean: if True returns a function to calculate
                          the transform. If False returns the Sympy
                          matrix
        """

        # check to see if we have our transformation saved in file
        if (os.path.isfile('%s/%s.T_inv' % (self.config_folder,
                                                name))):
            T_inv = cloudpickle.load(open('%s/%s.T_inv' %
                                          (self.config_folder, name), 'rb'))
        else:
            T = self._get_T(name=name)
            rotation_inv = T[:3, :3].T
            translation_inv = -rotation_inv * T[:3, 3]
            T_inv = rotation_inv.row_join(translation_inv).col_join(
                sp.Matrix([[0, 0, 0, 1]]))

            # save to file
            cloudpickle.dump(T_inv, open('%s/%s.T_inv' %
                                         (self.config_folder, name), 'wb'))

        if lambdify is False:
            return T_inv
        return sp.lambdify(self.q + self.x, T_inv)

# r = robot_config()
# q = [0.,0.,0.,0.,0.,0.]
# JEE = r.J('EE', q)
# dx = np.dot(JEE, np.array(q)+1)
# print dx
# r.Mq_g([0,0,0,0,0,0])
# r.Mq([0,0,0,0,0,0])
