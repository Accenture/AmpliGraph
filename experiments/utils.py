import numpy as np 
import logging
logging.basicConfig(level=logging.DEBUG)

def clean_data(train, valid, test, keep_valid = False):
    train_ent = set(train.flatten())
    valid_ent = set(valid.flatten())
    test_ent = set(test.flatten())
    
    if not keep_valid:
        # filter valid
        valid_diff_train_ent = valid_ent - train_ent
        idxs_valid = []
        if len(valid_diff_train_ent) > 0:
            count_valid = 0
            c_if = 0
            for row in valid:
                tmp = set(row)
                if len(tmp & valid_diff_train_ent) != 0:
                    idxs_valid.append(count_valid)
                    c_if+=1
                count_valid = count_valid + 1
        filtered_valid = np.delete(valid, idxs_valid, axis=0)
        logging.debug("shape valid: {0} shape - filtered valid: {1}: {2}".format(valid.shape, filtered_valid.shape, c_if))
        valid = filtered_valid
       
    # filter test 
    train_valid_ent = set(train.flatten()) | set(valid.flatten())
    ent_test_diff_train_valid = test_ent - train_valid_ent
    
    idxs_test = []

    if len(ent_test_diff_train_valid) > 0:
        count_test = 0
        c_if=0
        for row in test:
            tmp = set(row)
            if len(tmp & ent_test_diff_train_valid) != 0:
                idxs_test.append(count_test)
                c_if+=1
            count_test = count_test + 1
    filtered_test = np.delete(test, idxs_test, axis=0)
    logging.debug("shape test: {0} shape   -  filtered test: {1}: {2}".format(test.shape, filtered_test.shape, c_if))

    return valid, filtered_test