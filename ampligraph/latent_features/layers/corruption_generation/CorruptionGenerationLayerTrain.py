import tensorflow as tf


class CorruptionGenerationLayerTrain(tf.keras.layers.Layer):

    def __init__(self, eta, **kwargs):
        '''
        Initializes the corruption generation layer
        
        Parameters:
        -----------
        eta: int
            number of corruptions to generate
        '''
        self.eta = eta
        super(CorruptionGenerationLayerTrain, self).__init__(**kwargs)

    @tf.function
    def call(self, pos, ent_size):
        '''
        Generates corruption for the positives supplied 
        
        Parameters:
        -----------
        pos: (n, 3)
            batch of input triples (positives)
        ent_size:
            number of unique entities present in the partition
        
        Returns:
        --------
        corruptions: (n * eta, 3)
            corruptions of the triples
        '''
        # get the number of positives
        batch_size = tf.shape(pos)[0]
        # size and reshape the dataset to sample corruptions
        dataset = tf.reshape(tf.tile(tf.reshape(pos, [-1]), [self.eta]), [tf.shape(input=pos)[0] * self.eta, 3])
        # generate a mask which will tell which subject needs to be corrupted (random uniform sampling)
        keep_subj_mask = tf.cast(tf.random.uniform([tf.shape(input=dataset)[0]], 0, 2, dtype=tf.int32, seed=0), tf.bool)
        # If we are not corrupting the subject then corrupt the object
        keep_obj_mask = tf.logical_not(keep_subj_mask)
        
        # cast it to integer (0/1)
        keep_subj_mask = tf.cast(keep_subj_mask, tf.int32)
        keep_obj_mask = tf.cast(keep_obj_mask, tf.int32)
        # generate the n * eta replacements (uniformly randomly)
        replacements = tf.random.uniform([tf.shape(dataset)[0]], 0, ent_size, dtype=tf.int32, seed=0)
        # keep subjects of dataset where keep_subject is 1 and zero it where keep_subject is 0
        # now add replacements where keep_subject is 0 (i.e. keep_object is 1)
        subjects = tf.math.add(tf.math.multiply(keep_subj_mask, dataset[:, 0]),
                               tf.math.multiply(keep_obj_mask, replacements))
        # keep relations as it is
        relationships = dataset[:, 1]
        # keep objects of dataset where keep_object is 1 and zero it where keep_object is 0
        # now add replacements where keep_object is 0 (i.e. keep_subject is 1)
        objects = tf.math.add(tf.math.multiply(keep_obj_mask, dataset[:, 2]),
                              tf.math.multiply(keep_subj_mask, replacements))
        # stack the generated subject, reln and object entities and create the corruptions
        corruptions = tf.transpose(a=tf.stack([subjects, relationships, objects]))
        return corruptions
        

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
        return [batch_size * self.eta, 3]
