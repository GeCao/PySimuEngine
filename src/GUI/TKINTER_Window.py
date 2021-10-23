import sys
import traceback
import os
import time

import tkinter as tk

from ..utils.Constants import *

LOG_LINE_NUM = 0


class TKINTER_Window():
    def __init__(self, window):
        self.window = window
        self.width, self.height = global_screen_width,  global_screen_height

        self.menu = None
        self.log_data_Text = None

        self.initialized = False

    def initialization(self):
        self.initialized = True
        self.window.title("A Game Engine for Simulation / Rendering test")
        self.window.geometry(str(self.width) + 'x' + str(self.height) + '+10+10')
        self.window.resizable(width=True, height=True)
        self.window["bg"] = "black"
        # self.window.attributes('-transparentcolor', 'white')  # 使白色为透明色
        # self.window.attributes("-alpha", 0.5)                          #虚化，值越小虚化程度越高

        self.log_data_Text = tk.Text(self.window)  # 日志框
        self.log_data_Text.pack(fill="x", side="top")
        self.write_log_to_Text("INFO:Initialized Gui successfully!")

        # Set menu:
        self.menu = tk.Menu(self.window)
        self.menu.add_command(label="Exit", command=self.exit)

        # Set buttons；
        button = tk.Button(self.window, text="Exit")
        button.pack(fill="x", side="top")
        button.bind("<Button-1>", self.exit)

    def get_current_time(self):
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        return current_time

    def write_log_to_Text(self, logmsg):
        global LOG_LINE_NUM
        current_time = self.get_current_time()
        logmsg_in = str(current_time) + " " + str(logmsg) + "\n"  # 换行
        if LOG_LINE_NUM <= 7:
            self.log_data_Text.insert(tk.END, logmsg_in)
            LOG_LINE_NUM = LOG_LINE_NUM + 1
        else:
            self.log_data_Text.delete(1.0, 2.0)
            self.log_data_Text.insert(tk.END, logmsg_in)

    def exit(self, *args):
        self.window.quit()
        print("Interrupt by exit the window!")
        sys.exit()


def run_window(project_name):
    root = tk.Tk()
    tkinter_window = TKINTER_Window(root)
    tkinter_window.initialization()
    root.mainloop()
    sys.exit()
