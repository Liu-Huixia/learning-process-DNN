from __future__ import print_function
import tf_keras as keras
import tf_keras.backend as K
import numpy as np
from IPython.display import clear_output

from six.moves import cPickle
import os

import utils

class LoggingReporter(keras.callbacks.Callback):
    def __init__(self, cfg, trn, tst, do_save_func=None, *kargs, **kwargs):
        super(LoggingReporter, self).__init__(*kargs, **kwargs)
        self.cfg = cfg  # Configuration options dictionary
        self.trn = trn  # Train data
        self.tst = tst  # Test data
        
        if 'FULL_MI' not in cfg:
            self.cfg['FULL_MI'] = False # Whether to compute MI on train and test data, or just test
            
        if self.cfg['FULL_MI']:
            self.full = utils.construct_full_dataset(trn,tst)
        
        # do_save_func(epoch) should return True if we should save on that epoch
        self.do_save_func = do_save_func
      
    def on_train_begin(self, logs={}):
        if not os.path.exists(self.cfg['SAVE_DIR']):
            # print("Making directory", self.cfg['SAVE_DIR'])
            os.makedirs(self.cfg['SAVE_DIR'])
            
        # Indexes of the layers which we keep track of. Basically, this will be any layer 
        # which has a 'kernel' attribute, which is essentially the "Dense" or "Dense"-like layers
        self.layerixs = []
    
        # Functions return activity of each layer
        self.layerfuncs = []
        
        # Functions return weights of each layer
        self.layerweights = []

        for lndx, l in enumerate(self.model.layers):
            if hasattr(l, 'kernel'):
                self.layerixs.append(lndx)
                self.layerfuncs.append(K.function(self.model.inputs, [l.output,]))
                self.layerweights.append(l.kernel)



    def on_epoch_end(self, epoch, logs={}):
        if self.do_save_func is not None and not self.do_save_func(epoch):
            # Don't log this epoch
            return
        
        # Get overall performance
        loss = {}
        data = {
            'weights_norm' : [],   # L2 norm of weights
            'gradmean'     : [],   # Mean of gradients
            'gradstd'      : [],   # Std of gradients
            'activity_tst' : []    # Activity in each layer for test set
        }
        
        for lndx, layerix in enumerate(self.layerixs):
            # clayer = self.model.layers[layerix]
        
            # data['weights_norm'].append(np.linalg.norm(K.get_value(clayer.kernel)) )
            
            # stackedgrads = np.stack(self._batch_gradients[lndx], axis=1)
            # data['gradmean'    ].append(np.linalg.norm(stackedgrads.mean(axis=1)) )
            # data['gradstd'     ].append(np.linalg.norm(stackedgrads.std(axis=1)) )
            
            if self.cfg['FULL_MI']:
                data['activity_tst'].append(self.layerfuncs[lndx]([self.full.X,])[0])
            else:
                data['activity_tst'].append(self.layerfuncs[lndx]([self.tst.X,])[0])
        fname = self.cfg['SAVE_DIR'] + "/epoch%08d"% epoch
        
        with open(fname, 'wb') as f:
             cPickle.dump({'ACTIVATION':self.cfg['ACTIVATION'], 'epoch':epoch, 'data':data, 'loss':loss}, f, cPickle.HIGHEST_PROTOCOL)        
        
