import tensorly as tl
import numpy as np
def metricL1tucker(tensor, factors):
    core = tl.tenalg.multi_mode_dot(tensor, factors, modes = list(range(tensor.ndim)), transpose = True)
    core = core/tl.norm(core)
    #retensor = tl.tenalg.multi_mode_dot(core, factors, modes=list(range(tensor.ndim)), transpose=False)
    return np.sum(np.abs(core.flatten()))#np.sum(np.abs((tensor/tl.norm(tensor))-(retensor/tl.norm(retensor))))#np.sum(np.abs(core.flatten()))#np.sum(np.abs(tensor-retensor))#