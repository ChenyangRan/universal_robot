import cloudpickle
import os
import numpy as np
from sympy import sin, cos, pi, Matrix, Symbol, simplify, trigsimp, pprint, utilities, zeros
import sympy as sp


class robot_config:
    def __init__(self):
        self.num_joints = 6
        self.num_links = 6
        self.config_folder = 'ur5_config'

        # create function dictionaries
        self._Tx = {}  # for transform calculations
        self._T_inv = {}  # for inverse transform calculations
        self._J = {}  # for Jacobian calculations

        self._CoM = [] # center of mass of each joint
        self._M = []  # placeholder for (x,y,z) inertia matrices
        self._Mq = None  # placeholder for joint space inertia matrix function
        self._g = None  # placeholder for joint space gravity term function

        # set up our joint angle symbols
        self.q = [Symbol('q%i' % ii) for ii in range(self.num_joints)]
        self.dq = [Symbol('dq%i' % ii) for ii in range(self.num_joints)]
        # set up an (x,y,z) offset
        self.x = [Symbol('x'), Symbol('y'), Symbol('z')]

        self.gravity = Matrix([[0, 0, -9.81, 0, 0, 0]]).T

        self.link_names = ['link%i' % (ii) for ii in range(self.num_links)]
        self.q_names = ['joint%i' % (ii) for ii in range(self.num_links)]
        self.joint_names = ['UR5_joint%i' % (ii)
                            for ii in range(self.num_joints)]

        # for the null space controller, keep arm near these angles
        self.rest_angles = np.array([None,
                                     np.pi/4.0,
                                     -np.pi/2.0,
                                     np.pi/4.0,
                                     np.pi/2.0,
                                     np.pi/2.0])

        # create the inertia matrices for each link of the ur5
        # Diagonal matrix [m m m Ixx Iyy Izz]

        # self._M.append(np.diag([1.0, 1.0, 1.0, 0.02, 0.02, 0.02]))  # link0
        self._M.append(np.diag([2.5, 2.5, 2.5, 0.04, 0.04, 0.04]))  # link1
        self._M.append(np.diag([5.7, 5.7, 5.7, 0.06, 0.06, 0.04]))  # link2
        self._M.append(np.diag([3.9, 3.9, 3.9, 0.055, 0.055, 0.04]))  # link3
        self._M.append(np.copy(self._M[1]))  # link4
        self._M.append(np.copy(self._M[1]))  # link5
        self._M.append(np.diag([0.7, 0.7, 0.7, 0.01, 0.01, 0.01]))  # link6
        # self._M.append(np.diag(3*[3.7]  +[0.00167, 0.001762, 0.001238]))  # link1
        # self._M.append(np.diag(3*[8.393]+[0.08478, 0.0874, 0.009851]))  # link2
        # self._M.append(np.diag(3*[2.33]+[0.0597, 0.06091, 0.0005315]))  # link3
        # self._M.append(np.diag(3*[1.219]+[0.007461, 0.007624, 0.004633]))  # link4
        # self._M.append(np.diag(3*[1.219]+[0.004669, 0.007785, 0.0076]))  # link5
        # self._M.append(np.diag(3*[0.1879]+[0.003870, 0.002621, 0.002621]))  # link6 

        self._CoM.append([0.0,  -0.02, 0])
        self._CoM.append([0.13,  0.0,  0.1157])
        self._CoM.append([0.05,  0.0,  0.0238])
        self._CoM.append([0.0,   0.0,  0.01])
        self._CoM.append([0.0,   0.0,  0.01])
        self._CoM.append([0.0,   0.0,  -0.02])
        self._CoM.append([0.0,   0.0,  0.0]) ## end effector

        self._calc_T()

        Jw = Matrix([0,0,1])
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

    def Jx(self, name, q, x=[0, 0, 0]):
        """ Calculates the transform for a joint or link

        name string: name of the joint or link, or end-effector
        q np.array: joint angles
        """
        # check for function in dictionary
        if self._J.get(name, None) is None:
            print('Generating Jacobian function for %s' % name)
            self._J[name] = self._calc_Jx(name, x=x)
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

    def Mqx(self, q):
        """ Calculates the joint space inertia matrix for the ur5

        q np.array: joint angles
        """
        # check for function in dictionary
        if self._Mq is None:
            print('Generating inertia matrix function')
            self._Mq = self._calc_Mq(jacobianx=True)
        parameters = tuple(q) + (0, 0, 0)
        return np.array(self._Mq(*parameters))

    def _calc_Mq(self, lambdify=True, jacobianx=False):
        """ Uses Sympy to generate the inertia matrix in
        joint space for the ur5

        lambdify boolean: if True returns a function to calculate
                          the Jacobian. If False returns the Sympy
                          matrix
        """
        filename = 'Mqx' if jacobianx else 'Mq' 
        filename = '%s/%s' % (self.config_folder, filename)
        # check to see if we have our Jacobian saved in file
        Mq = self._get_from_file(filename)
        if Mq is None:
            # get the Jacobians for each link's COM
            if jacobianx:
                J = [self._calc_Jx('link%s' % (ii), lambdify=False)
                    for ii in range(self.num_links)] 
            else:
                J = [self._calc_J('link%s' % (ii), lambdify=False)
                    for ii in range(self.num_links)] 
            

            # transform each inertia matrix into joint space
            # sum together the effects of arm segments' inertia on each motor
            Mq = zeros(self.num_joints)
            for ii in range(self.num_links):
                Mq += Matrix(J[ii].T) * Matrix(self._M[ii]) * Matrix(J[ii])
            Mq = Matrix(Mq)

            if lambdify:
                Mq = sp.lambdify(self.q + self.x, Mq)

            # save to file
            self._save_to_file(filename, Mq)

        return Mq

    def g(self, q):
        """ Calculates the force of gravity in joint space for the ur5

        q np.array: joint angles
        """
        # check for function in dictionary
        if self._g is None:
            print('Generating gravity effects function')
            self._g = self._calc_g()
        parameters = tuple(q) + (0, 0, 0)
        return np.array(self._g(*parameters)).flatten()

    def gx(self, q):
        """ Calculates the force of gravity in joint space for the ur5

        q np.array: joint angles
        """
        # check for function in dictionary
        if self._g is None:
            print('Generating gravity effects function')
            self._g = self._calc_g(jacobianx=True)
        parameters = tuple(q) + (0, 0, 0)
        return np.array(self._g(*parameters)).flatten()

    def _calc_g(self, lambdify=True, jacobianx=False):
        """ Generate the force of gravity in joint space

        Uses Sympy to generate the force of gravity in joint space

        Parameters
        ----------
        lambdify : boolean, optional (Default: True)
            if True returns a function to calculate the matrix.
            If False returns the Sympy matrix
        """
        filename = 'gx' if jacobianx else 'g' 
        filename = '%s/%s' % (self.config_folder, filename)
        # check to see if we have our gravity term saved in file
        g = self._get_from_file(filename)

        if g is None:
            # if no saved file was loaded, generate function
            print('Generating gravity compensation function')
            get_J = self._calc_Jx if jacobianx else self._calc_J
            # get the Jacobians for each link's COM
            J_links = [get_J('link%s' % (ii), x=[0, 0, 0],
                                    lambdify=False)
                       for ii in range(self.num_links)]

            # sum together the effects of each arm segment's inertia
            g = zeros(self.num_joints, 1)
            for ii in range(self.num_joints):
                # transform each inertia matrix into joint space
                g += (J_links[ii].T * self._M[ii] * self.gravity)
            g = Matrix(g)

            if lambdify:
                g = sp.lambdify(self.q + self.x, g)

            # save to file
            self._save_to_file(filename, g)

        return g

    def _calc_J(self, name, x=[0,0,0], lambdify=True):
        """ Uses Sympy to generate the Jacobian for a joint or link

        name string: name of the joint or link, or end-effector
        lambdify boolean: if True returns a function to calculate
                          the Jacobian. If False returns the Sympy
                          matrix
        """
        filename = '%s/%s.J' % (self.config_folder, name)
        # check to see if we have our Jacobian saved in file
        J = self._get_from_file(filename)
        if J is None:
            Tx = self._calc_Tx(name, x=x, lambdify=False)
            J = []

            # calculate derivative of (x,y,z) wrt to each joint
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
            self._save_to_file(filename, J)

        J = Matrix(J).T  # correct the orientation of J
        if lambdify is False:
            return J
        return sp.lambdify(self.q + self.x, J)

    def _calc_Jx(self, name, x=[0,0,0], lambdify=True):
        """ Uses Sympy to generate the Jacobian for a joint or link
        Non-derivative approach

        name string: name of the joint or link, or end-effector
        lambdify boolean: if True returns a function to calculate
                          the Jacobian. If False returns the Sympy
                          matrix
        """
        filename = '%s/%s.Jx' % (self.config_folder, name)
        # check to see if we have our Jacobian saved in file
        J = self._get_from_file(filename)
        if J is None:
            link = int(name.strip('link').strip('joint'))-1
            CoM = Matrix(self._calc_Tx(name, x, False)[0:3]).T
            J = []
            Jv = []
            Jw = []
            Jv.append(Matrix([0,0,1]).cross(CoM).T.tolist())
            Jw.append(Matrix([0,0,1]).tolist())
            for i in xrange(1,link+1):
                Z = trigsimp(Matrix(self._get_T('link%s'%(i))[2,0:3]))
                P = trigsimp(Matrix(self._get_T('link%s'%(i))[3,0:3]))
                Jv.append(trigsimp(Z.cross(CoM - P).tolist()))
                Jw.append(Z.tolist())
            n = len(Jv)
            
            for i in range(n):
                J.append(Jv[i]+Jw[i])  
            J = utilities.iterables.flatten(J)    
            J = J + (self.num_links**2 - len(J))*[0]
            J = Matrix(J).reshape(6,6).T

            # save to file
            self._save_to_file(filename, J)
        if lambdify is False:
            return J
        return sp.lambdify(self.q + self.x, J)

    def _calc_T(self):
        # segment lengths associated with each joint
        dh = [0.089159,-0.425,-0.39225,0.10915,0.09465,0.0823]

        self.T01 =             self._compute_dh_matrix(0.,     pi/2,    dh[0],  self.q[0]-pi/2)
        self.T02 =  self.T01 * self._compute_dh_matrix(dh[1],    0.,       0.,  self.q[1]-pi/2)
        self.T03 =  self.T02 * self._compute_dh_matrix(dh[2],    0.,       0.,  self.q[2])
        self.T04 =  self.T03 * self._compute_dh_matrix(0.,      pi/2,   dh[3],  self.q[3]-pi/2)
        self.T05 =  self.T04 * self._compute_dh_matrix(0.,     -pi/2,   dh[4],  self.q[4])
        self.T06 =  self.T05 * self._compute_dh_matrix(0.,        0.,   dh[5],  self.q[5])
        self.T0EE = self.T06
        self.T = [self.T01,self.T02,self.T03,self.T04,self.T05,self.T06,self.T0EE]
        self.T = simplify(self.T)


    def _compute_dh_matrix(self, r, alpha, d, theta):
        A = [[cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha),   r*cos(theta)],
             [sin(theta),  cos(theta)*cos(alpha),  -cos(theta)*sin(alpha),  r*sin(theta)],
             [      0,               sin(alpha),                cos(alpha),       d     ],
             [      0,               0,                        0,                 1     ]]
        return Matrix(A)

    def _get_T(self, name):
        if name == "EE":
            T = self.T[self.link_names.index('link5')]
        else:
            try:
                T = self.T[self.link_names.index(name)] 
            except:
                T = self.T[self.q_names.index(name)] 
        return T

    def _calc_Tx(self, name, x=[0,0,0], lambdify=True):
        """ Uses Sympy to transform x from the reference frame of a joint
        or link to the origin (world) coordinates.

        name string: name of the joint or link, or end-effector
        x list: the [x,y,z] position of interest in "name"'s reference frame
        lambdify boolean: if True returns a function to calculate
                          the transform. If False returns the Sympy
                          matrix
        """

        # check to see if we have our transformation saved in file
        filename = '%s/%s.T_inv' % (self.config_folder, name)
        T_inv = self._get_from_file(filename)
        if T_inv is None:
            T = self._get_T(name)
            if 'link' in name:
                _link = int(name.strip('link'))-1
                x = (np.array(x) + np.array(self._CoM[_link])).tolist()
            # transform x into world coordinates
            Tx = T * Matrix(x + [1])
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
        filename = '%s/%s.T_inv' % (self.config_folder, name)
        T_inv = self._get_from_file(filename)
        # check to see if we have our transformation saved in file
        if T_inv is None:
            T = self._get_T(name=name)
            rotation_inv = T[:3, :3].T
            translation_inv = -rotation_inv * T[:3, 3]
            T_inv = rotation_inv.row_join(translation_inv).col_join(
                Matrix([[0, 0, 0, 1]]))

            # save to file
            self._save_to_file(filename, T_inv)

        if lambdify is False:
            return T_inv
        return sp.lambdify(self.q + self.x, T_inv)

    def _get_from_file(self, filename):
        try:
            expr = cloudpickle.load(open(filename, 'rb'))
        except Exception:
            expr = None
        return expr

    def _save_to_file(self, filename, data):
        # save to file
        cloudpickle.dump(data, open(filename, 'wb'))    

# r = robot_config()
# pprint(r._calc_Jx('link2',[0,0,0], False))
# print "#######################"
# J = r._calc_J('link2',[0,0,0], lambdify=False)
# pprint(J)
# q = [0.,0.,0.,0.,0.,0.]
# JEE = r.J('EE', q)
# dx = np.dot(JEE, np.array(q)+1)
# print dx
# r.Mq_g([0,0,0,0,0,0])
# r.Mq([0,0,0,0,0,0])
