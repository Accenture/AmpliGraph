import tensorflow as tf
import abc

LOSS_REGISTRY = {}


def register_loss(name, external_params=[], class_params= {'require_same_size_pos_neg' : True,}):
    def insert_in_registry(class_handle):
        LOSS_REGISTRY[name] = class_handle
        class_handle.name = name
        LOSS_REGISTRY[name].external_params = external_params
        LOSS_REGISTRY[name].class_params = class_params
        return class_handle
        
    return insert_in_registry


class Loss(abc.ABC):
    """Abstract class for loss function.
    """
    
    name = ""
    external_params = []
    class_params = {}
    
    def __init__(self, eta, hyperparam_dict, verbose=False):
        """Initialize Loss

        Parameters
        ----------
        eta: int
            number of negatives
        hyperparam_dict : dict
            dictionary of hyperparams for the loss
        """
        self._loss_parameters = {}
        self._dependencies = []
        
        #perform check to see if all the required external hyperparams are passed
        try:
            self._loss_parameters['eta'] = eta
            self._init_hyperparams(hyperparam_dict)
            if verbose:
                print('------ Loss-----')
                print('Name:', self.name)
                print('Parameters:')
                for key in self._loss_parameters.keys():
                    print("  ", key, ": ", self._loss_parameters[key])
            
        except KeyError:
            raise Exception('Some of the hyperparams for loss were not passed to the loss function')
            
    def get_state(self, param_name):
        """Get the state value

        Parameters
        ----------
        param_name : string
            name of the state for which one wants to query the value
        Returns
        -------
        param_value:
            the value of the corresponding state
        """
        try:
            param_value = LOSS_REGISTRY[self.name].class_params.get(param_name)
            return param_value
        except KeyError:
             raise Exception('Invald Key')

    def _init_hyperparams(self, hyperparam_dict):
        """ Verifies and stores the hyperparams needed by the algorithm
        
        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The Loss will check the keys to get the corresponding params
        """
        NotImplementedError("This function is a placeholder in an abstract class")
        
    def _inputs_check(self, scores_pos, scores_neg):
        """ Creates any dependencies that need to be checked before performing loss computations
        
        Parameters
        ----------
        scores_pos : tf.Tensor
            A tensor of scores assigned to positive statements.
        scores_neg : tf.Tensor
            A tensor of scores assigned to negative statements.
        """
        self._dependencies = []
        if LOSS_REGISTRY[self.name].class_params['require_same_size_pos_neg'] and self._loss_parameters['eta']!=1:
             self._dependencies.append(tf.Assert(tf.equal(tf.shape(scores_pos)[0] ,tf.shape(scores_neg)[0]), [tf.shape(scores_pos)[0], tf.shape(scores_neg)[0]]))
        

        
    def _apply(self, scores_pos, scores_neg):
        """ Apply the loss function. Every inherited class must implement this function. (All the TF code must go in this function.)
        
        Parameters
        ----------
        scores_pos : tf.Tensor
            A tensor of scores assigned to positive statements.
        scores_neg : tf.Tensor
            A tensor of scores assigned to negative statements.
        
        Returns
        -------
        loss : float
            The loss value that must be minimized.
        """
        NotImplementedError("This function is a placeholder in an abstract class")
        
    def apply(self, scores_pos, scores_neg):
        """ Interface to external world. This function does the input checks, preprocesses input and finally applies loss fn.
        Parameters
        ----------
        scores_pos : tf.Tensor
            A tensor of scores assigned to positive statements.
        scores_neg : tf.Tensor
            A tensor of scores assigned to negative statements.
        
        Returns
        -------
        loss : float
            The loss value that must be minimized.
        """
        self._inputs_check(scores_pos, scores_neg)
        with tf.control_dependencies(self._dependencies):
            loss = self._apply(scores_pos, scores_neg)
        return loss

@register_loss("pairwise", ['margin'])
class PairwiseLoss(Loss):
    """Pairwise, max-margin loss.

     Introduced in :cite:`bordes2013translating`.

    .. math::

        \mathcal{L}(\Theta) = \sum_{t^+ \in \mathcal{G}}\sum_{t^- \in \mathcal{C}}max(0, [\gamma + f_{model}(t^-;\Theta) - f_{model}(t^+;\Theta)])

    where :math:`\gamma` is the margin, :math:`\mathcal{G}` is the set of positives,
    :math:`\mathcal{C}` is the set of corruptions, :math:`f_{model}(t;\Theta)` is the model-specific scoring function.

    """
    
    def __init__(self, eta, hyperparam_dict, verbose=False):
        super().__init__(eta,  hyperparam_dict, verbose)
        
    def _init_hyperparams(self, hyperparam_dict):
        """ Verifies and stores the hyperparams needed by the algorithm
        
        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The Loss will check the keys to get the corresponding params('margin')
        """
        self._loss_parameters['margin'] = hyperparam_dict.get('margin', 1)
        
            
    def _apply(self, scores_pos, scores_neg):
        margin = tf.constant(self._loss_parameters['margin'], dtype=tf.float32, name='margin')
        loss = tf.reduce_sum(tf.maximum(margin- scores_pos + scores_neg, 0))
        return loss
    

@register_loss("nll")        
class NLLLoss(Loss):
    """Negative log-likelihood loss.

    As described in :cite:`trouillon2016complex`.

    .. math::

        \mathcal{L}(\Theta) = \sum_{t \in \mathcal{G} \cup \mathcal{C}}log(1 + exp(-yf_{model}(t;\Theta)))

    where :math:`y` is the label of the statement :math:` \in [-1, 1]`, :math:`\mathcal{G}` is the set of positives,
    :math:`\mathcal{C}` is the set of corruptions, :math:`f_{model}(t;\Theta)` is the model-specific scoring function.

    """
    def __init__(self, eta, hyperparam_dict, verbose=False):
        super().__init__(eta, hyperparam_dict, verbose)
    
    def _init_hyperparams(self, hyperparam_dict):
        """ Verifies and stores the hyperparams needed by the algorithm
        
        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The Loss will check the keys to get the corresponding params
        """
        return
        
    def _apply(self, scores_pos, scores_neg):
        """ Apply the loss function
        Parameters
        ----------
        scores_pos : tf.Tensor, shape [n, 1]
            A tensor of scores assigned to positive statements.
        scores_neg : tf.Tensor, shape [n, 1]
            A tensor of scores assigned to negative statements.

        Returns
        -------
        loss : float
            The loss value that must be minimized.

        """
        scores = tf.concat([-scores_pos, scores_neg], 0)
        return tf.reduce_sum(tf.log(1 + tf.exp(scores)))

    
@register_loss("absolute_margin", ['margin'] )      
class AbsoluteMarginLoss(Loss):
    """Absolute margin , max-margin loss.

        Introduced in :cite:`Hamaguchi2017`.

       .. math::

           \mathcal{L}(\Theta) = \sum_{t^+ \in \mathcal{G}}\sum_{t^- \in \mathcal{C}} f_{model}(t^-;\Theta) - max(0, [\gamma - f_{model}(t^+;\Theta)])

       where :math:`\gamma` is the margin, :math:`\mathcal{G}` is the set of positives,
       :math:`\mathcal{C}` is the set of corruptions, :math:`f_{model}(t;\Theta)` is the model-specific scoring function.
    """
    
    def __init__(self, eta, hyperparam_dict, verbose=False):
        super().__init__(eta, hyperparam_dict, verbose)
        
    def _init_hyperparams(self, hyperparam_dict):
        """ Verifies and stores the hyperparams needed by the algorithm
        
        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The Loss will check the keys to get the corresponding params('margin')
        """
        self._loss_parameters['margin'] =hyperparam_dict.get('margin', 1)
        
    
    def _apply(self, scores_pos, scores_neg):
        """ Apply the loss function
        Parameters
        ----------
        scores_pos : tf.Tensor, shape [n, 1]
           A tensor of scores assigned to positive statements.
        scores_neg : tf.Tensor, shape [n, 1]
           A tensor of scores assigned to negative statements.

        Returns
        -------
        loss : float
           The loss value that must be minimized.

        """
        margin =  tf.constant(self._loss_parameters['margin'], dtype=tf.float32, name='margin')
        loss = tf.reduce_sum(scores_neg + tf.maximum(margin - scores_pos, 0))
        return loss
    

@register_loss("self_adverserial", ['margin', 'alpha'], {'require_same_size_pos_neg':False})      
class SelfAdverserialLoss(Loss):
    """Self adverserial sampling loss

        Introduced in :cite:`rotatE`.

       .. math::

           \mathcal{L} = -log \sigma(\gamma - d_r (h,t)) - \sum_{i=1}^{n} p(h_{i}^{'} ,r,t_{i}^{'} ) log \sigma(d_r (h_{i}^{'},t_{i}^{'}) - \gamma)

       where :math:`\gamma` is the margin, and p(h_{i}^{'} ,r,t_{i}^{'} ) is the sampling proportion
    """
    def __init__(self, eta, hyperparam_dict, verbose=False):
        super().__init__(eta, hyperparam_dict, verbose)
    
    def _init_hyperparams(self, hyperparam_dict):
        """ Verifies and stores the hyperparams needed by the algorithm
        
        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The Loss will check the keys to get the corresponding params('margin')
        """
        self._loss_parameters['margin'] = hyperparam_dict.get('margin', 3)
        self._loss_parameters['alpha'] = hyperparam_dict.get('alpha', 0.5)
        
    
    
    def _apply(self, scores_pos, scores_neg):
        """ Apply the loss function
       Parameters
       ----------
       scores_pos : tf.Tensor, shape [n, 1]
           A tensor of scores assigned to positive statements.
       scores_neg : tf.Tensor, shape [n*negative_count, 1]
           A tensor of scores assigned to negative statements.

       Returns
       -------
       loss : float
           The loss value that must be minimized.

       """
        margin = tf.constant(self._loss_parameters['margin'], dtype=tf.float32, name='margin')
        alpha = tf.constant(self._loss_parameters['alpha'], dtype=tf.float32, name='alpha')
    
        #Compute p(neg_samples) based on eq 4
        scores_neg_reshaped = tf.reshape(scores_neg, [self._loss_parameters['eta'], tf.shape(scores_pos)[0]])
        p_neg = tf.nn.softmax(alpha * scores_neg_reshaped, axis = 0)
        #Compute Loss based on eg 5
        loss = tf.reduce_sum(-tf.log( tf.nn.sigmoid(margin -  tf.negative(scores_pos)) )) - \
                                tf.reduce_sum(tf.multiply(p_neg, \
                                                          tf.log( tf.nn.sigmoid( tf.negative(scores_neg_reshaped) - margin)) ))
        return loss
