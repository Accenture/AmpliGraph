import tensorflow as tf


class EmbeddingLookupLayer(tf.keras.layers.Layer):

    def __init__(self, max_ent_size, max_rel_size, k, **kwargs):
        '''
        Initializes the embeddings of the model
        
        Parameters:
        -----------
        max_ent_size: int
            max entities that can occur in any partition
        max_rel_size: int
            max relations that can occur in any partition
        k: int
            embedding size
        seed: int 
            random seed
        '''
        self.max_ent_size = max_ent_size
        self.max_rel_size = max_rel_size
        self.k = k

        # create the trainable variables for entity embeddings
        self.ent_emb = tf.Variable(tf.initializers.GlorotUniform()(shape=[self.max_ent_size, self.k]), 
                                   name='ent_emb_layer_1', trainable=True)
        
        # create the trainable variables for relation embeddings
        self.rel_emb = tf.Variable(tf.initializers.GlorotUniform()(shape=[self.max_rel_size, self.k]), 
                                   name='rel_emb_layer_1', trainable=True)
        
        super(EmbeddingLookupLayer, self).__init__(**kwargs)
        
    @tf.function
    def partition_change_updates(self, batch_ent_emb, batch_rel_emb):
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
        paddings_ent = tf.constant([[0, self.max_ent_size - batch_ent_emb.shape[0]], [0, 0]])
        paddings_rel = tf.constant([[0, self.max_rel_size - batch_rel_emb.shape[0]], [0, 0]])

        # once padded, assign it to the trainable variable
        self.ent_emb.assign(tf.pad(batch_ent_emb, paddings_ent, 'CONSTANT', constant_values=0))
        self.rel_emb.assign(tf.pad(batch_rel_emb, paddings_rel, 'CONSTANT', constant_values=0))
        
    @tf.function
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
