import os
import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        depth = d_model / 2

        positions = np.arange(max_len)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth   # (1, depth)

        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)

        pos_encoding = np.concatenate(
          [np.sin(angle_rads), np.cos(angle_rads)], axis=-1
        )

        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[tf.newaxis, :seq_len, :]


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(d_model),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        attn_output = self.self_attn(query=x, value=x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)


class TripleEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(max_len=4, d_model=d_model)
        self.encoder_block = TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout)
        self.projection = tf.keras.layers.Dense(latent_dim)  # project to latent space

        # v0.2: we try to prepend a CLS to the triple and use it to extract the latent representation
        self.cls_token = self.add_weight(
            shape=(1, 1, d_model),  # (batch_size=1, token=1, embedding_dim)
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )

    def call(self, x, training=False, mask=None):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])  # (batch_size, 1, d_model)

        # Prepend CLS token to the sequence
        x = tf.concat([cls_tokens, x], axis=1)  # (batch_size, 4, d_model)

        x = self.pos_encoder(x)              # Add position info
        x = self.encoder_block(x, training=training, mask=mask)

        cls_output = x[:, 0, :]
        latent = self.projection(cls_output) # (batch_size * num_triples, latent_dim)
        return latent


class FFNNDecoder(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.ffnn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(output_dim)  # same as input embedding dim
        ])

    def call(self, x):
        return self.ffnn(x)


class TripleModel(tf.keras.Model):
    def __init__(
            self, d_model, num_heads, ff_dim,
            latent_dim, decoder_hidden_dim,
            embedding_dim, ent_emb, rel_emb,
            loss="cosine"):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = TripleEncoder(d_model, num_heads, ff_dim, latent_dim)
        self.decoder = FFNNDecoder(decoder_hidden_dim, embedding_dim)

        self.ent_emb = ent_emb
        self.rel_emb = rel_emb

        self.loss_tracker = tf.keras.metrics.Mean()
        self.loss_fn = loss

    def call(self, x, training=False):
        """
        Parameters
        ----------
        x: tensor
            The tensor of batches of triples that to be encoded in a unique representation.
            The shape of this tensor is (batch_size, num_triples, 3, pretrained_embedding_dim).
        training: bool
            Whether we are training (`True`) or performing inference (`False`).
        """
        # Extract a mask for the padded triples in the batch
        mask = tf.reduce_all(
            x != -1, axis=-1, keepdims=True
        )
        # We extract the embeddings for the entities
        x = self.get_input_embeddings(x, mask=mask)

        # We first reshape triples to the shape needed by the MultiHeadAttention layer.
        batch_size, num_triples, _, _ = x.shape
        x = tf.reshape(x, (-1, 3, x.shape[-1]))

        encoded_emb = self.encoder(x, training=training, mask=tf.reshape(mask, (-1, 1, 1)))

        # encoded_emb shape: (batch_size * num_triples, latent_dim)
        # mask shape: (batch_size * num_triples, 1)
        # We average the triples to get the unique latent representation of the target entity
        latent = tf.math.segment_mean(
            tf.boolean_mask(encoded_emb, tf.reshape(mask, [-1])),
            segment_ids=self.extract_segment_ids(tf.squeeze(mask, axis=-1))
        )

        # We decode the latent representation
        output = self.decoder(latent, training=training)

        return output


    def get_input_embeddings(self, triples, mask=None):
        """
        Given batches of triples, retrieve and return the embeddings.
        Parameters
        ----------
        triples: tensor
            Tensor of shape (batch_size, num_triples, 3) that stores the batches of triples.
        Returns
        -------
        tensor
            Tensor of shape (batch_size, num_triples, 3, embedding_dim) where embedding_dim
            is the dimension of the pretrained KGE embeddings.
        """
        # First of all, we set indices == -1 to zero, to avoid errors with tf.gather
        if mask is not None:
            binary_mask = tf.cast(mask, tf.int32)
            triples = triples * binary_mask

        # Get components of triples
        subjs = triples[..., 0]
        rels = triples[..., 1]
        objs = triples[..., 2]

        subj_embs = tf.gather(self.ent_emb, subjs)  # (batch, triples, k)
        rel_embs = tf.gather(self.rel_emb, rels)
        obj_embs = tf.gather(self.ent_emb, objs)

        # No need to mask the embeddings, because they are not weights, so they will not
        # affect gradient computations. This should change in case we implement a joint
        # optimization of KGE embeddings and encoder-decoder.

        return tf.stack([subj_embs, rel_embs, obj_embs], axis=2)

    def train_step(self, data):

        triples, target = data
        # We extract the KGE embeddings of the triples and target
        target_emb = tf.gather(self.ent_emb, target)

        with tf.GradientTape() as tape:
            # We encode and decode the triples to get a unique representation for each of batches.
            output_emb = self(triples, training=True)

            # Normalize both for cosine similarity
            output_emb_norm = tf.math.l2_normalize(output_emb, axis=-1)
            target_emb_norm = tf.math.l2_normalize(target_emb, axis=-1)

            # Cosine similarity loss = 1 - cosine_sim
            if self.loss_fn == "cosine":
                cos_sim = tf.reduce_mean(output_emb_norm * target_emb_norm, axis=-1)
                loss = 1.0 - cos_sim
            elif self.loss_fn == "euclidean":
                loss = tf.norm(output_emb_norm - target_emb_norm, axis=-1)

            # Apply gradients
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

            self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(shape=(None, None, 3), dtype=tf.int32)]
    # )
    # def evaluate(self, test_set):
    #     """
    #     Function to perform inference.
    #     """
    #     for el in test_set:
    #         self(el, training=False)


    def extract_segment_ids(self, mask):
        """
        Given the mask used to select the triples to consider when computing the
        latent representation, this function assigns the index of each batch to the
        values to keep in the batch and then flatten the arrays.

        This function is needed to compute the masked average of embeddings through
        tf.math.segment_mean.
        """
        batch_size = tf.shape(mask)[0]

        # Create a tensor of shape (batch_size, 1) with row indices
        row_indices = tf.range(batch_size, dtype=tf.int32)[:, tf.newaxis]

        # Broadcast to same shape as mask
        broadcasted = tf.broadcast_to(row_indices, tf.shape(mask))

        return tf.ragged.boolean_mask(broadcasted, mask).flat_values
