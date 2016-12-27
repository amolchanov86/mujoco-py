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

    def viewerSetup(self):
        self.width = 640*2
        self.height = 480*2
        self.viewer = mjviewer.MjViewer(visible=True,
                                        init_width=self.width,
                                        init_height=self.height)
        # self.viewer.cam.trackbodyid = 0 #2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[0] = 0  # 0.8
        self.viewer.cam.lookat[1] = 0.5  # 0.8
        self.viewer.cam.lookat[2] = 0.1  # 0.8
        self.viewer.cam.elevation = 160
        self.viewer.cam.azimuth = 100
        # self.viewer.cam.pose =
        self.viewer.cam.camid = -3
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


if __name__ == "__main__":
    myBox = tableScenario()
    myBox.viewerSetup()
    saveData = False
    myBox.viewer = myBox.viewerStart()
    #force_pos =  myBox.model.geom_pose('cBox_1')[0] + np.array([-0.04, 0.02, 0.02])
    #box_pos = myBox.model.data.xipos[1]
    #f_direction = (box_pos - force_pos)/np.linalg.norm(box_pos-force_pos)
    f_direction = np.random.randn(3)
    force = 500*f_direction
    torque = np.random.randn(3)*0.00
    myBox.model.data.qfrc_applied = np.hstack([force, torque])
    for j in range(10000):
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