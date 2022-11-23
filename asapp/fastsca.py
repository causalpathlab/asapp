from data import _loader
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

class FASTSCA:
    def __init__(self):
        self.config = None
        self.data = self.data()
    
    class data:
        def __init__(self):
            self.mtx_indptr = None
            self.mtx_indices = None
            self.mtx_data = None
            self.rows = None
            self.cols = None
            self.mtx = None

    def initdata(self):
        _loader.initialize_data(self)

    def loaddata(self):
        _loader.load_data(self)

