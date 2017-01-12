import os
import mujoco_py
from mujoco_py import mjviewer, mjcore
from mujoco_py import mjtypes
from mujoco_py import glfw
import numpy as np
import ctypes
from os import path
from os.path import dirname, abspath
import six
from math import *
import random
import transforms3d
from PIL import Image, ImageDraw
import time
import psutil

counter = 1
def image(image, point):
    global counter
    data, width, height = image
    x, y = point
    print point
    img = Image.new('RGB', (width, height))
    img.frombytes(data)
    draw = ImageDraw.Draw(img)
    draw.ellipse((x, y, x+2, y+2), fill='blue', outline='blue')
    draw.ellipse((191, 28, 197, 35), fill='red')
    img.save('test/image-{}.png'.format(counter))
    counter += 1

model_folder_path = dirname(dirname(abspath(__file__))) + '/examples/models/'


class tableScenario():
    def __init__(self):
        self.xml_path = model_folder_path + 'blocks_gen.xml'
        if not path.exists(self.xml_path):
            raise IOError("File %s does not exist" % self.xml_path)
        self.model = mjcore.MjModel(self.xml_path)
        self.dt = self.model.opt.timestep;
        # self.action_space = spaces.Box(self.lower, self.upper)
        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': int(np.round(1.0 / self.dt))}
        print 'Initialization finished ...'

    def viewerSetup(self):
        self.width = 640*2
        self.height = 480*2
        self.viewer = mjviewer.MjViewer(visible=True,
                                        init_width=self.width,
                                        init_height=self.height)
        # self.viewer.cam.trackbodyid = 0 #2
        self.viewer.cam.distance = 1.56
        # self.viewer.cam.lookat[0] = 0  # 0.8
        # self.viewer.cam.lookat[1] = 0.5  # 0.8
        # self.viewer.cam.lookat[2] = 0.1  # 0.8
        self.viewer.cam.elevation = -90.00
        self.viewer.cam.azimuth = 90.33
        # self.viewer.cam.pose =
        # self.viewer.cam.camid = -3
        self.viewer.start()
        self.viewer.set_model(self.model)
        # (data, width, height) = self.viewer.get_image()

    def viewerEnd(self):
        self.viewer.finish()
        self.viewer = None

    def viewerStart(self):
        if self.viewer is None:
            self.viewerSetup()
        return self.viewer

    def viewerRender(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewerStart().finish()
                self.viewer = None
            return
        if mode == 'rgb_array':
            self.viewerStart().render()
            self.viewerStart().set_model(self.model)
            data, width, height = self.viewerStart().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        elif mode == 'human':
            self.viewerStart().loop_once()

    def resetModel(self):
        self.model.resetData()
        ob = self.resetModel()
        if self.viewer is not None:
            self.viewer.autoscale()
            self.viewerSetup()
        return ob

    def getComPos(self):
        ridx = self.model.body_names.index(six.b(body_name))
        return self.model.data.com_subtree[idx]

    def getComVel(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.body_comvels[idx]

    def getXmat(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.data.xmat[idx].reshape((3, 3))

    def getStateVector(self, model):
        return np.concatenate([model.data.qpos.flat,
                               model.data.qvel.flat])

    def setPos(self, model, q):
        # print model.data.qpos
        model.data.qpos = q
        model._compute_subtree()
        model.forward()
        # print model.data.qpos
        return model

    def setVel(self, model, dq):
        model.data.qvel = dq
        model._compute_subtree()
        model.forward()
        return model

    def setControl(self, model, ctrl, nFrames=1):
        model.data.ctrl = ctrl
        for _ in range(nFrames):
            model.step()
        return model

    def applyFTOnObj(self, point_name):
        point = self.model.geom_pose(point_name)
        com = self.model.data.xipos[1]
        f_direction = (com - point) / np.linalg.norm(com - point)
        torque = np.random.randn(3) * 0.0
        self.model.data.qfrc_applied = self.model.applyFT(point, force, torque, 'object')

    def resetBox(self):
        self.setPos(myBox.model, np.array([0.25, 0., 0.15, 1., 0., 0., 0.]))
        self.setVel(myBox.model, np.zeros(6))
        self.qfrc_applied = np.zeros((6, 1))
        self.model.forward()

    def applyFTOnObj2(self, point_name1, point_name2):
        point1 = self.model.geom_pose(point_name1)[0]
        point2 = self.model.geom_pose(point_name2)[0]
        com = self.model.data.xipos[1]
        f_direction1 = (com - point1) / np.linalg.norm(com - point1)
        f_direction2 = (com - point2) / np.linalg.norm(com - point2)
        force1 = 500. * f_direction1  # 500*np.random.randn(3)#500*f_direction
        force2 = 500. * f_direction2  # 500*np.random.randn(3)#500*f_direction
        torque = np.random.randn(3) * 0.0
        self.model.data.qfrc_applied = np.hstack([force1 + force2, torque])
        # self.model.applyFT(point,force,torque,'object')
        return f_direction1, f_direction2

    def applyFTOnObj1(self, point_name):
        point = self.model.geom_pose(point_name)[0]
        com = self.model.data.xipos[1]
        f_direction = (com - point) / np.linalg.norm(com - point)
        force = 5. * f_direction  # 500*np.random.randn(3)#500*f_direction
        torque = np.random.randn(3) * 0.0
        self.model.data.qfrc_applied = np.hstack([force, torque])
        # self.model.applyFT(point,force,torque,'object')
        return f_direction

    def find_relative_config_change(self, start_state, next_state):
        change_pos = next_state[0:3] - start_state[0:3]
        change_ori = next_state[3] * start_state[4:6] - start_state[3] * next_state[4:6] + np.cross(next_state[4:6],
                                                                                                    start_state[4:6])
        return change_pos, change_ori

    def find_distances_among_forces(self, point_name1, point_name2):
        point1 = self.model.geom_pose(point_name1)[0]
        point1_face = self.model.geom_pose(point_name1[0:2])[0]
        point2 = self.model.geom_pose(point_name2)[0]
        point2_face = self.model.geom_pose(point_name2[0:2])[0]
        dist_pt1_pt2 = np.linalg.norm(point1 - point2)
        dist_pt1_f = np.linalg.norm(point1 - point1_face)
        dist_pt2_f = np.linalg.norm(point2 - point2_face)
        return dist_pt1_f, dist_pt2_f, dist_pt1_pt2

def local_coordinates(point, com, orientation):
    # print 'default orientation:', orientation
    # print 'rotation matrix:', transforms3d.quaternions.quat2mat(orientation)
    # print 'point:', point
    # print 'com:', com
    rotation_matrix = transforms3d.quaternions.quat2mat(orientation)
    point_n = np.vstack([np.reshape(point,(3,1)),[1]])
    com_n = np.reshape(com, (3,1))
    T =  np.vstack([np.hstack([rotation_matrix, com_n]), np.reshape([0,0,0,1], (1,4))])
    T_inv =  np.linalg.inv(T)
    # print 'T_inv:', T_inv
    # print np.dot(T_inv,point_n), np.dot(T, np.dot(T_inv,point_n))
    return np.reshape(np.dot(T_inv,point_n), (4,))[:-1]

def check_contact(point, coms, orientations):
    '''
    Check the contact with the geoms
    :param point:
    :param coms: Center of mass of blocks
    :param orientations: orientation of blocks in quaternions
    :return: name of the block
    '''
    for i,com in enumerate(coms):
        new_cor_point = local_coordinates(point, com, orientations[i])
        if all([abs(val)<=0.02 for val in new_cor_point]):
            return 'custom_object_{}'.format(i+1)
    return None

def test_contact_force():
    print 'Contact test start ...'
    myBox = tableScenario()
    myBox.viewerSetup()
    myBox.viewer = myBox.viewerStart()
    for x in np.arange(-0.6, 0.6, 0.04 ):
        for y in np.arange(-0.6, 0.6, 0.04):
            myBox.model.data.qfrc_applied = np.hstack([np.zeros(3), np.zeros(3)])
            box = check_contact(np.array([x,y,0.03]), myBox.model.data.xpos[1:], myBox.model.data.xquat[1:])
            image(myBox.viewer.get_image(), (1279 / 2.0 + 790 * x, 963 / 2.0 + 790 * y))
            if box is not None:
                # print box, myBox.model.body_pose(box) if box is not None else None
                com = myBox.model.body_pose(box)[0]
                point =  np.array([x, y, 0.036])
                f_direction = (com-point)/np.linalg.norm(com-point)

                print 'com = ', com
                force = 500*f_direction
                torque = np.random.rand(3)
                # myBox.model.data.qfrc_applied = myBox.model.applyFT(point, force, torque, 'custom_object_{}'.format(box.split("_")[-1]))
                # myBox.model.data.qfrc_applied = myBox.model.applyFT(point, force, torque,
                #                                                     'custom_object_{}'.format(box.split("_")[-1]))
                point = com
                body_name = 'custom_object_1'
                mjcore.applyFT(point, force, torque, body_name)
            myBox.model.step()
            myBox.viewerRender()

    myBox.viewerEnd()
    print 'Contact test end ...'

def test_contact_force2():
    print 'Contact test start ...'
    myBox = tableScenario()
    myBox.viewerSetup()
    myBox.viewer = myBox.viewerStart()
    for x in np.arange(-0.6, 0.6, 0.04 ):
        for y in np.arange(-0.6, 0.6, 0.04):
            myBox.model.data.qfrc_applied = np.hstack([np.zeros(3), np.zeros(3)])
            # box = check_contact(np.array([x,y,0.03]), myBox.model.data.xpos[1:], myBox.model.data.xquat[1:])
            body_name = 'custom_object_3'
            body_adr = myBox.model.body_name2id(body_name)
            print 'body addr = ', body_adr, ' addr_type = ', type(body_adr)
            image(myBox.viewer.get_image(), (1279 / 2.0 + 790 * x, 963 / 2.0 + 790 * y))
            if body_name is not None:
                # print box, myBox.model.body_pose(box) if box is not None else None
                com = myBox.model.body_pose(body_name)[0]
                #point =  np.array([x, y, 0.036])
                #point = com - np.array([0.1, 0, 0])
                # f_direction = (com-point)/np.linalg.norm(com-point)
                # f_direction = 500*np.random.rand(3)
                f_direction = 500*np.array([0.,1.,0.])

                force = f_direction
                print 'com = ', com, ' direction = ', f_direction, ' force = ', force
                torque = np.random.rand(3) * 0
                # torque = np.ones([1,3])
                # myBox.model.data.qfrc_applied = myBox.model.applyFT(point, force, torque, 'custom_object_{}'.format(box.split("_")[-1]))
                # myBox.model.data.qfrc_applied = myBox.model.applyFT(point, force, torque,
                #                                                     'custom_object_{}'.format(box.split("_")[-1]))
                point = com - np.array([0.001, 0, 0])
                myBox.model.applyFT(point, force, torque, body_name)

            myBox.model.step()
            myBox.viewerRender()

    myBox.viewerEnd()
    print 'Contact test end ...'



if __name__ == "__main__":
    test_contact_force2()
    exit()
    myBox = tableScenario2()
    myBox.viewerSetup()
    saveData = False
    myBox.viewer = myBox.viewerStart()
    #test_contact_force()
    #force_pos =  myBox.model.geom_pose('cBox_1')[0] + np.array([-0.04, 0.02, 0.02])
    #box_pos = myBox.model.data.xipos[1]
    #f_direction = (box_pos - force_pos)/np.linalg.norm(box_pos-force_pos)
    f_direction = np.random.randn(3)
    force = 500*f_direction
    torque = np.random.randn(3)*0.00
    myBox.model.data.qfrc_applied = np.hstack([force, torque])
    for j in range(10000):
        #print local_coordinates(np.random.randn(3), myBox.model.data.geom_xpos[1], myBox.model.data.xquat[1])
        # print myBox.model.data.geom_xpos, myBox.model.data.xquat
        # f_direction = np.array([12.4000000e-01,   2.77555756e-17,   4.00000000e-02])
        # print check_contact(f_direction, myBox.model.data.xpos[1:], myBox.model.data.xquat[1:])

        if j%500==0:
            f_direction = np.random.randn(3)
            force = 500*f_direction
            torque = np.random.randn(3) * 0.00
            myBox.model.data.qfrc_applied = np.hstack([force, torque])
        if (j-1)%500==0:
            myBox.model.data.qfrc_applied = np.hstack([np.zeros(len(force)), np.zeros(len(torque))])
        start_state = myBox.getStateVector(myBox.model)
        myBox.viewerRender()
        myBox.model.step()
        next_state = myBox.getStateVector(myBox.model)
    myBox.viewerEnd()