###############################################################################
# 
# Utility functions for loading and saving the compressed model.
#
###############################################################################

import pickle
import bz2


# Compresses and saves the model.
# Note: Use '.pbz2' extension for the compressed file.
def compress_pickle(filename, data):
    with bz2.BZ2File(filename, 'w') as f: 
        pickle.dump(data, f)

# Loads and decompresses the compressed model.
def decompress_pickle(filename):
    with bz2.BZ2File(filename, 'rb') as f:
        data = pickle.load(f)
    return data
