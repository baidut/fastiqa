from .bunch import *

############# log #############################################################
import sys
from loguru import logger

DEBUG = False

if DEBUG:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
else:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
