import sys
import os

import numpy as np
import pygame
from pygame.locals import *
import copy
from ..utils.utils import *
# from ..utils.Constants import *

class MyPygame:
    def __init__(self, core_component):
        self.core_component = core_component
        width, height = global_screen_width, global_screen_height
        self.size = width, height  # 设置窗口大小

        self.screen = None
        self.mouse_clicked = False
        self.mouse_pos = np.array([0.0, 0.0])

        self.is_running = False
        self.initialized = False

    def initialization(self):
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        pygame.mixer.pre_init(44100, -16, 2, 4096)
        pygame.mixer.init()
        pygame.init()
        self.screen_width = pygame.display.Info().current_w
        self.screen_height = pygame.display.Info().current_h
        self.screen = pygame.display.set_mode(self.size, pygame.OPENGL)  # 显示窗口

        self.is_running = True
        self.initialized = True

    def run(self):
        if self.is_running:
            for event in pygame.event.get():
                self.exec_event(event)

    def exec_event(self, event):
        eye, center, eye_up = copy.deepcopy(self.core_component.camera.get_look_at_params())
        if event.type == QUIT:
            print("Interrupt by quit the PyGame window!")
            self.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_UP:
                print("KeyUp introduced!")
                eye[2] += 0.2
                self.core_component.camera.update_mvp(eye=eye)
            if event.key == K_DOWN:
                print("KeyDown introduced!")
                eye[2] -= 0.2
                self.core_component.camera.update_mvp(eye=eye)
            if event.key == K_RIGHT:
                print("KetRight introduced!")
                eye[1] += 0.2
                self.core_component.camera.update_mvp(eye=eye)
            if event.key == K_LEFT:
                print("KeyLeft introduced!")
                eye[1] -= 0.2
                self.core_component.camera.update_mvp(eye=eye)
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1 or 3:
                self.mouse_clicked = True
                self.mouse_pos = np.array(pygame.mouse.get_pos())
        elif event.type == MOUSEBUTTONUP:
            self.mouse_clicked = False

        if self.mouse_clicked and event.type == MOUSEMOTION:
            new_mouse_pos = np.array(pygame.mouse.get_pos())
            move_vec = new_mouse_pos - self.mouse_pos
            anticlockwise = [True if move_vec[0] > 0 else False, True if move_vec[0] > 0 else False]
            self.mouse_pos = new_mouse_pos
            move_vec[0] /= global_screen_width * 0.01
            move_vec[1] /= global_screen_height * 0.01
            move_angle_unit = [move_vec[0] if anticlockwise[0] else -move_vec[0],
                               move_vec[1] if anticlockwise[1] else -move_vec[1]]
            eye = get_rotation_mat_from_euler_angle(cross(eye - center, eye_up), move_angle_unit[1]).dot(eye)
            eye = get_rotation_mat_from_euler_angle(eye_up, move_angle_unit[0]).dot(eye)
            self.core_component.camera.update_mvp(eye=eye)

    def quit(self):
        pygame.display.quit()
        pygame.quit()
        self.is_running = False