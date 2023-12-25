import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '../..')
autograd_path = os.path.abspath(parent_dir)
sys.path.append(autograd_path)

from autograd.engine import Value
import autograd.nn as nn

X = [[1, 2, 3], [2, 3, 4], [-3, 4, -2.3], [-5, 6, -2]]
y = [0, 1, 0, 1]

n = Net()