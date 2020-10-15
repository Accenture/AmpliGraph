import tensorflow as tf
import numpy as np


class EmbeddingLookupLayer(tf.keras.layers.Layer):

    def __init__(self, k, max_ent_size=None, max_rel_size=None, 
                 entity_kernel_initializer="glorot_uniform", entity_kernel_regularizer=None, 
                 relation_kernel_initializer="glorot_uniform", relation_kernel_regularizer=None, 
                 **kwargs):
        '''
        Initializes the embeddings of the model
        
        Parameters:
        -----------
        k: int
            embedding size
        max_ent_size: int
            max entities that can occur in any partition (default: None)
        max_rel_size: int
            max relations that can occur in any partition (default: None)
        entity_kernel_initializer: String (name of objective function), objective function or 
            `tf.keras.initializers.Initializer` instance
            An objective function is any callable with the signature `init = fn(shape)`
            Initializer of the entity embeddings.
        entity_kernel_regularizer: String (name of objective function), objective function or 
            `tf.keras.initializers.Initializer` instance
            An objective function is any callable with the signature `init = fn(shape)`
            Initializer of the relation embeddings.
        relation_kernel_initializer: String (name of objective function), objective function or 
            `tf.keras.regularizers.Regularizer` instance
            Regularizer of entity embeddings.
        relation_kernel_regularizer: String (name of objective function), objective function or 
            `tf.keras.regularizers.Regularizer` instance
            Regularizer of relations embeddings.
        seed: int 
            random seed
        '''
        super(EmbeddingLookupLayer, self).__init__(**kwargs)
        self._has_enough_args_to_build_ent_emb = False
        self._has_enough_args_to_build_rel_emb = False
        self._max_ent_size_internal = None
        self._max_rel_size_internal = None
        self.k = k

        self.ent_partition = None
        self.rel_partition = None

        self.max_ent_size = max_ent_size
        self.max_rel_size = max_rel_size
        
        self.ent_init = entity_kernel_initializer
        self.rel_init = relation_kernel_initializer
        
        self.ent_regularizer = entity_kernel_regularizer
        self.rel_regularizer = relation_kernel_regularizer
        
    def set_ent_rel_initial_value(self, ent_init, rel_init):
        ''' Sets the initial value of entity and relation embedding matrix.
        This function is mainly used during the partitioned training where the full embedding matrix is 
        initialized outside the model.
        '''
        self.ent_partition = ent_init
        self.rel_partition = rel_init
        
    def set_initializer(self, initializer):
        ''' Set the initializer of the weights of this layer
        
        Parameters:
        -----------
        initializer: String (name of objective function), objective function or 
            `tf.keras.initializers.Initializer` instance
            An objective function is any callable with the signature `init = fn(shape)`
            Initializer of the entity and relation embeddings. This is either a single value or a list of size 2.
            If it is a single value, then both the entities and relations will be initialized based on 
            the same initializer. if it is a list, the first initializer will be used for entities and second 
            for relations.
        '''
        if isinstance(initializer, list):
            assert len(initializer) == 2, \
                'Incorrect length for initializer. Assumed 2 got {}'.format(len(initializer))
            self.ent_init = tf.keras.initializers.get(initializer[0])
            self.rel_init = tf.keras.initializers.get(initializer[1])
        else:
            self.ent_init = tf.keras.initializers.get(initializer)
            self.rel_init = tf.keras.initializers.get(initializer)
            
    def set_regularizer(self, regularizer):
        ''' Set the regularizer of the weights of this layer
        
        Parameters:
        -----------
        regularizer: String (name of objective function), objective function or 
            `tf.keras.regularizers.Regularizer` instance
            Regularizer of entities and relations.
            If it is a single value, then both the entities and relations will be regularized based on 
            the same regularizer. if it is a list, the first regularizer will be used for entities and second 
            for relations.
        '''
            
        if isinstance(regularizer, list):
            assert len(regularizer) == 2, \
                'Incorrect length for regularizer. Assumed 2 got {}'.format(len(regularizer))
            self.ent_regularizer = tf.keras.regularizers.get(regularizer[0])
            self.rel_regularizer = tf.keras.regularizers.get(regularizer[1])
        else:
            self.ent_regularizer = tf.keras.regularizers.get(regularizer)
            self.rel_regularizer = tf.keras.regularizers.get(regularizer)
        
    @property 
    def max_ent_size(self):
        ''' returns the value of size of entity embedding matrix
        '''
        return self._max_ent_size_internal
    
    @max_ent_size.setter 
    def max_ent_size(self, value):
        ''' Setter for the max entity size property.
        The layer is buildable only if this property is set
        '''
        if value is not None and value > 0:
            self._max_ent_size_internal = value
            self._has_enough_args_to_build_ent_emb = True
    
    @property 
    def max_rel_size(self):
        ''' returns the value of size of relation embedding matrix
        '''
        return self._max_rel_size_internal
    
    @max_rel_size.setter 
    def max_rel_size(self, value):
        ''' Setter for the max relation size property.
        The layer is buildable only if this property is set
        '''
        if value is not None and value > 0:
            self._max_rel_size_internal = value
            self._has_enough_args_to_build_rel_emb = True
    
    def build(self, input_shape):
        '''Builds the embedding lookup error. The trainable weights are created based on the hyperparams.
        '''
        # create the trainable variables for entity embeddings
        if self._has_enough_args_to_build_ent_emb: 
            self.ent_emb = self.add_weight(
                'ent_emb',
                shape=[self._max_ent_size_internal, self.k],
                initializer=self.ent_init,
                regularizer=self.ent_regularizer,
                dtype=tf.float32,
                trainable=True)

            if self.ent_partition is not None:
                paddings_ent = [[0, self._max_ent_size_internal - self.ent_partition.shape[0]], [0, 0]]
                self.ent_emb.assign(np.pad(self.ent_partition, paddings_ent, 'constant', constant_values=0))
                del self.ent_partition
                self.ent_partition = None
        
        else:
            raise TypeError('Not enough arguments to build Encoding Layer. Please set max_ent_size property.')
            
        # create the trainable variables for relation embeddings
        if self._has_enough_args_to_build_rel_emb: 
            self.rel_emb = self.add_weight(
                'rel_emb',
                shape=[self._max_rel_size_internal, self.k],
                initializer=self.rel_init,
                regularizer=self.rel_regularizer,
                dtype=tf.float32,
                trainable=True)
            
            if self.rel_partition is not None:
                paddings_rel = [[0, self._max_rel_size_internal - self.rel_partition.shape[0]], [0, 0]]
                self.rel_emb.assign(np.pad(self.rel_partition, paddings_rel, 'constant', constant_values=0))
                del self.rel_partition
                self.rel_partition = None
        else:
            raise TypeError('Not enough arguments to build Encoding Layer. Please set max_rel_size property.')
        
    @tf.function
    def partition_change_updates(self, partition_ent_emb, partition_rel_emb):
        ''' perform the changes that are required when the partition is changed during training
        
        Parameters:
        -----------
        batch_ent_emb:
            entity embeddings that need to be trained for the partition 
            (all triples of the partition will have embeddings in this matrix)
        batch_rel_emb:
            relation embeddings that need to be trained for the partition 
            (all triples of the partition will have embeddings in this matrix)
        
        '''
        
        # if the number of entities in the partition are less than the required size of the embedding matrix
        # pad it. This is needed because the trainable variable size cant change dynamically. 
        # Once defined, it stays fixed. Hence padding is needed.
        paddings_ent = tf.constant([[0, self._max_ent_size_internal - partition_ent_emb.shape[0]], [0, 0]])
        paddings_rel = tf.constant([[0, self._max_rel_size_internal - partition_rel_emb.shape[0]], [0, 0]])

        # once padded, assign it to the trainable variable
        self.ent_emb.assign(tf.pad(partition_ent_emb, paddings_ent, 'CONSTANT', constant_values=0))
        self.rel_emb.assign(tf.pad(partition_rel_emb, paddings_rel, 'CONSTANT', constant_values=0))
        
    @tf.function(experimental_relax_shapes=True)
    def call(self, triples):
        '''
        Looks up the embeddings of entities and relations of the triples
        
        Parameters:
        -----------
        triples: (n, 3)
            batch of input triples
        
        Returns:
        --------
        list: 
            list of embeddings of subjects, predicates, objects.
        '''
        # look up in the respective embedding matrix
        e_s = tf.nn.embedding_lookup(self.ent_emb, triples[:, 0])
        e_p = tf.nn.embedding_lookup(self.rel_emb, triples[:, 1])
        e_o = tf.nn.embedding_lookup(self.ent_emb, triples[:, 2])
        return [e_s, e_p, e_o]

    def compute_output_shape(self, input_shape):
        ''' returns the output shape of outputs of call function
        
        Parameters:
        -----------
        input_shape: 
            shape of inputs of call function
        
        Returns:
        --------
        output_shape:
            shape of outputs of call function
        '''
        assert isinstance(input_shape, list)
        batch_size, _ = input_shape
        return [(batch_size, self.k), (batch_size, self.k), (batch_size, self.k)]   
