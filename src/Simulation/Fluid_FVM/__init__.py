import os
import sys
Simulation_path = os.path.abspath(os.path.dirname(os.getcwd()))
src_path = os.path.abspath(os.path.dirname(Simulation_path))
main_path = os.path.abspath(os.path.dirname(src_path))
sys.path.append(main_path)