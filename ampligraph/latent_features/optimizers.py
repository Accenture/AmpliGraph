import tensorflow as tf
import abc
import logging

import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Default learning rate for the optimizers
DEFAULT_LR = 0.0005

# Default momentum for the optimizers
DEFAULT_MOMENTUM = 0.9

DEFAULT_DECAY_CYCLE = 0

DEFAULT_DECAY_CYCLE_MULTIPLE = 1

DEFAULT_LR_DECAY_FACTOR = 2

DEFAULT_END_LR = 0.00000001

DEFAULT_SINE = False

class DefaultOptimizer(abc.ABC):
    """Abstract class for loss function.
    """

    def __init__(self, optimizer, hyperparam_dict, batches_count):
        """Initialize the Optimizer
        
        Parameters
        ----------
        optimizer: string
            name of the optimizer to use
        hyperparam_dict : dict
            dictionary of hyperparams that would be used by the optimizer.
        batches_count: int
            number of batches in an epoch
        """
        
        self.optimizer = optimizer  
        self.optimizer_params = hyperparam_dict
        self.batches_count = batches_count


    def minimize(self, loss):
        """Create an optimizer to minimize the model loss 
        
        Parameters
        ----------
        loss: tf.Tensor
            Node which needs to be evaluated for computing the model loss.
            
        Returns
        -------
        train: tf.Operation
            Node that needs to be evaluated for minimizing the loss during training
        """
        if optimizer == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.optimizer_params.get('lr', DEFAULT_LR))
        elif optimizer == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.optimizer_params.get('lr', DEFAULT_LR))
        elif optimizer == "momentum":
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.optimizer_params.get('lr', DEFAULT_LR),
                                                        momentum=self.optimizer_params.get('momentum',
                                                                                           DEFAULT_MOMENTUM))
            logger.info('Momentum : {}'.format(self.optimizer_params.get('momentum', DEFAULT_MOMENTUM)))
            
        train = self.optimizer.minimize(loss)
        return train
        
        
    def update_feed_dict(self, feed_dict, batch_num, epoch_num):
        """Fills values of placeholders created by the optimizers.
        
        Parameters
        ----------
        feed_dict : dict
            Dictionary that would be passed while optimizing the model loss to sess.run.
        batch_num: int
            current batch number
        epoch_num: int
            current epoch number
        """
        return
        
class SGDOptimizer(DefaultOptimizer):

    def __init__(self, optimizer, hyperparam_dict, batches_count):
        """Initialize the Optimizer
        
        Parameters
        ----------
        optimizer: string
            name of the optimizer to use
        hyperparam_dict : dict
            dictionary of hyperparams that would be used by the optimizer.
        """
        super(SGDOptimizer, self).__init__(optimizer, hyperparam_dict, batches_count)


    def minimize(self, loss):
        """Create an optimizer to minimize the model loss 
        
        Parameters
        ----------
        loss: tf.Tensor
            Node which needs to be evaluated for computing the model loss.
            
        Returns
        -------
        train: tf.Operation
            Node that needs to be evaluated for minimizing the loss during training
        """
        
        #create a placeholder for learning rate
        self.lr_placeholder = tf.placeholder(tf.float32)
        #create the optimizer with the placeholder
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr_placeholder)
        
        #load the hyperparameters that would be used while generating the learning rate per batch
        #start learning rate
        self.start_lr = self.optimizer_params.get('lr', DEFAULT_LR)
        self.current_lr = self.start_lr
        
        #cycle rate for learning rate decay
        self.decay_cycle_rate = self.optimizer_params.get('decay_cycle', DEFAULT_DECAY_CYCLE)
        self.end_lr = self.optimizer_params.get('end_lr', DEFAULT_END_LR)
        
        #check if it is a sinudoidal decay or constant decay
        self.is_sine_decay = self.optimizer_params.get('sine_decay', DEFAULT_SINE)
        self.next_cycle_epoch = self.decay_cycle_rate
        
        #Get the cycle expand factor
        self.decay_cycle_expand_factor = self.optimizer_params.get('expand_factor', DEFAULT_DECAY_CYCLE_MULTIPLE)
        
        #Get the LR decay factor at the start of each cycle
        self.decay_lr_rate = self.optimizer_params.get('decay_lr_rate', DEFAULT_LR_DECAY_FACTOR)
        self.curr_cycle_length = self.decay_cycle_rate
        self.curr_start = 0
        
        #create the operation that minimizes the loss
        train = self.optimizer.minimize(loss)
        return train
        
        
    def update_feed_dict(self, feed_dict, batch_num, epoch_num):
        """Fills values of placeholders created by the optimizers.
        
        Parameters
        ----------
        feed_dict : dict
            Dictionary that would be passed while optimizing the model loss to sess.run.
        batch_num: int
            current batch number
        epoch_num: int
            current epoch number
        """
        #Sinusoidal Decay
        if self.is_sine_decay:
            #compute the cycle number
            current_cycle_num = ((epoch_num-1 - self.curr_start)  + 
                         (batch_num) / (1.0 * self.batches_count))/self.curr_cycle_length
            
            #compute a learning rate for the current batch/epoch
            self.current_lr = self.end_lr + (self.start_lr - self.end_lr) * \
                                0.5 *(1 + math.cos(math.pi * current_cycle_num))

            #Start the next cycle and Expand the cycle/Decay the learning rate
            if epoch_num % self.next_cycle_epoch==0 and batch_num==self.batches_count:
                self.curr_cycle_length = self.curr_cycle_length * self.decay_cycle_expand_factor
                self.next_cycle_epoch = self.next_cycle_epoch + self.curr_cycle_length
                self.curr_start = epoch_num
                self.start_lr = self.start_lr / self.decay_lr_rate 
        
        #fixed rate decay
        elif self.decay_cycle_rate>0:
            if epoch_num%self.next_cycle_epoch == 0 and batch_num==1:
                if self.current_lr > self.end_lr:
                    self.next_cycle_epoch = self.decay_cycle_rate + self.next_cycle_epoch * self.decay_cycle_expand_factor
                    self.current_lr = self.current_lr / self.decay_lr_rate
                    
                    if self.current_lr < self.end_lr:
                        self.current_lr = self.end_lr
                        
        #no change to the learning rate                 
        else:
            pass
                
        feed_dict.update({self.lr_placeholder : self.current_lr})
        