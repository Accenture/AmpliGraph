#######################################################################################################
# If not specified, following defaults will be used at respective locations

DEFAULT_INITIALIZER = 'xavier'

# Default learning rate for the optimizers
DEFAULT_LR = 0.0005

# Default momentum for the optimizers
DEFAULT_MOMENTUM = 0.9

# Default burn in for early stopping
DEFAULT_BURN_IN_EARLY_STOPPING = 100

# Default check interval for early stopping
DEFAULT_CHECK_INTERVAL_EARLY_STOPPING = 10

# Default stop interval for early stopping
DEFAULT_STOP_INTERVAL_EARLY_STOPPING = 3

# default evaluation criteria for early stopping
DEFAULT_CRITERIA_EARLY_STOPPING = 'mrr'

# default value which indicates whether to normalize the embeddings after each batch update
DEFAULT_NORMALIZE_EMBEDDINGS = False

# Default side to corrupt for evaluation
DEFAULT_CORRUPT_SIDE_EVAL = 's,o'

# default hyperparameter for transE
DEFAULT_NORM_TRANSE = 1

# default value for the way in which the corruptions are to be generated while training/testing.
# Uses all entities
DEFAULT_CORRUPTION_ENTITIES = 'all'

# Threshold (on number of unique entities) to categorize the data as Huge Dataset (to warn user)
ENTITY_WARN_THRESHOLD = 5e5

# Default value for k (embedding size)
DEFAULT_EMBEDDING_SIZE = 100

# Default value for eta (number of corrputions to be generated for training)
DEFAULT_ETA = 2

# Default value for number of epochs
DEFAULT_EPOCH = 100

# Default value for batch count
DEFAULT_BATCH_COUNT = 100

# Default value for seed
DEFAULT_SEED = 0

# Default value for optimizer
DEFAULT_OPTIM = "adam"

# Default value for loss type
DEFAULT_LOSS = "nll"

# Default value for regularizer type
DEFAULT_REGULARIZER = None

# Default value for verbose
DEFAULT_VERBOSE = False

# Specifies how to generate corruptions for training - default does s and o together and applies the loss
DEFAULT_CORRUPT_SIDE_TRAIN = ['s,o']

# Subject corruption with a OneToNDatasetAdapter requires an N*N matrix (where N is number of unique entities).
# Specify a batch size to reduce memory overhead.
DEFAULT_SUBJECT_CORRUPTION_BATCH_SIZE = 10000

# Default hyperparameters for ConvEmodel
DEFAULT_CONVE_CONV_FILTERS = 32
DEFAULT_CONVE_KERNEL_SIZE = 3
DEFAULT_CONVE_DROPOUT_EMBED = 0.2
DEFAULT_CONVE_DROPOUT_CONV = 0.3
DEFAULT_CONVE_DROPOUT_DENSE = 0.2
DEFAULT_CONVE_USE_BIAS = True
DEFAULT_CONVE_USE_BATCHNORM = True
