import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '../..')
autograd_path = os.path.abspath(parent_dir)
sys.path.append(autograd_path)

from autograd.engine import Value
import autograd.nn as nn

train_x = [[]]