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
        print 'viewerSetup finished...'

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

def test_contact_force2(create_images=False):
    print 'Contact test start ...'
    myBox = tableScenario()
    myBox.viewerSetup()
    myBox.viewer = myBox.viewerStart()
    for x in np.arange(-0.6, 0.6, 0.04 ):
        for y in np.arange(-0.6, 0.6, 0.04):
            myBox.model.data.qfrc_applied = np.zeros((myBox.model.nv,1), dtype=np.double)
            body_name = check_contact(np.array([x,y,0.03]), myBox.model.data.xpos[1:], myBox.model.data.xquat[1:])
            if create_images: image(myBox.viewer.get_image(), (1279 / 2.0 + 790 * x, 963 / 2.0 + 790 * y))
            if body_name is not None:
                print '\n------------{}-----------'.format(body_name)
                com = myBox.model.body_pose(body_name)[0]
                force = 50*np.array([0.,1.,0.])
                torque = np.array([0.,0.,0.])

                print 'com = ', com, 'force = ', force, 'torque = ', torque
                point = com - np.array([0., 0., 0.])
                myBox.model.applyFT(point, force, torque, body_name)
                t = 10
                while t:
                    myBox.model.step()
                    t -= 1
            myBox.model.step()
            myBox.viewerRender()

    myBox.viewerEnd()
    print 'Contact test end ...'



if __name__ == "__main__":
    test_contact_force2()
