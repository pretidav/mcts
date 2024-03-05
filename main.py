import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import coloredlogs
import warnings
from connect4.Connect4Game import Connect4Game as Game

from network.nn import Connect4NNet
import tensorflow as tf 
log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.
if __name__=='__main__':
    g = Game(height=4,width=4, win_length=3)
    log.info('Loading %s...', Game.__name__)
    # log.info(g._base_board)
    # log.info(Connect4NNet(game=g,args={}))