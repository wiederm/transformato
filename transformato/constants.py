import logging, sys
from simtk import unit

logger = logging.getLogger(__name__)

platform = "CPU"
temperature = 303.15 * unit.kelvin
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
kT = kB * temperature

this = sys.modules[__name__]
# we can explicitly make assignments on it
this.NUM_PROC = 1


def initialize_NUM_PROC(n_proc):
    if this.NUM_PROC == 1:
        # also in local function scope. no scope specifier like global is needed
        this.NUM_PROC = n_proc
    else:
        msg = "NUM_PROC is already initialized to {0}."
        raise RuntimeError(msg.format(this.NUM_PROC))


def check_platform(configuration: dict):

    if platform.upper() == "GPU":
        configuration["simulation"]["GPU"] = True
        print("Setting platform to GPU")
    elif platform.upper() == "CPU":
        configuration["simulation"]["GPU"] = False
        print("Setting platform to CPU")
    else:
        raise RuntimeError("something went wrong")
