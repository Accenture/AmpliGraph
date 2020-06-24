
import sqlite3
from sqlite3 import Error
import numpy as np
from urllib.request import pathname2url
import os
import shelve
import pandas as pd
from datetime import datetime
from ampligraph.utils.profiling import get_human_readable_size

def load_csv(data_source, chunk_size=None, sep='\t', verbose=False):
    data = pd.read_csv(data_source, sep=sep, chunksize=chunk_size)
    if verbose:
        print("data type:", type(data))
        print("CSV loaded, into iterator data.")
        
    if isinstance(data, pd.DataFrame):
        return data.values
    else:
        return data

def load_gz(data_source, verbose=False):
    raise NotImplementedError
    
def load_tar(data_source, verbose=False):
    raise NotImplementedError

class DataSourceIdentifier():
    def __init__(self, data_source, verbose=False):
        self.verbose = verbose
        self.data_source = data_source
        self.supported_types = {"csv": load_csv, 
                                "txt": load_csv, 
                                "gz": load_gz, 
                                "tar": load_tar}
        self._identify()
                
    def fetch_loader(self):
        if self.verbose:
            print("Return adequate loader that provides loading of data source.")
        return self.supported_types[self.src]
    
    def _identify(self):
        if isinstance(self.data_source, str):
            self.src =  self.data_source.split(".")[-1] if "." in self.data_source else None           
            if self.src is not None and self.src not in self.supported_types:
                print("File type not supported! Supported types: {}".format(", ".join(self.supported_types)))
                self.src = None   
