"""
Project: Game Engine. Will be used for simulation, rendering, and optimization, as well.
Produced by: Ge Cao (gecaoge@student.ethz.ch)
"""

import sys
import os
import time
from multiprocessing import Process
from src.GUI.TKINTER_Window import run_window
from src.Components.Core_Component import CoreComponent

import OpenGL
OpenGL.ERROR_CHECKING = True
OpenGL.ERROR_LOGGING = True
OpenGL.FULL_LOGGING = False
OpenGL.ALLOW_NUMPY_SCALARS = True


def run(project_name):
    # editor_process = Process(target=run_window, args=(project_name,))
    # editor_process.start()

    core_component = CoreComponent()
    core_component.initialization()
    core_component.run()


if __name__ == "__main__":
    project_name = "Debug"
    run(project_name)