import pytest
from ampligraph.compat import TransE
import numpy as np
from ampligraph.explanations import ExamplE
from ampligraph.explanations import Decoder

SCOPE = "function"

embeddings = {
    'patient534': np.array([1,100]), 
    'patient1234': np.array([0,100]),
    'patient98': np.array([1,99]),    
    'patient721': np.array([20,80]), 
    'patient25': np.array([30,70]), 
    
    'hasrelapse': np.array([50,50]), 
    'hasProgression': np.array([48,50]),      
    'hasCardiacArrest': np.array([48,47]),  
    
    'relapse': np.array([100,0]),
    'progression': np.array([99,1]),
    'CardiacArrest': np.array([99,2]),   
    
    'currentCancertype': np.array([1,1]), 
    'lung': np.array([1,2]),
    'birthDate': np.array([3,5]), 
    '1955-03-04': np.array([5,3]),
    'hasSmokingHistory': np.array([19,22]), 
    'SmokingHistory': np.array([34,9]),
    'hasBiomarker': np.array([33,2]), 
    'ROS1': np.array([9,11]),    
    'hasBiomarker': np.array([98,99]), 
    'ALK': np.array([100,97]), 
    'hasComorbidity': np.array([89,89]), 
    'Dyslipidemia': np.array([90,80]),  
    'Cardiopathy': np.array([79,98]),       
    'therapytype': np.array([78,78]),
    'PalliativeCare': np.array([89,92])    
}


train_triples = np.array([
    ['patient534', 'hasrelapse', 'relapse'],
    ['patient534', 'currentCancertype', 'lung'],
    ['patient534', 'birthDate', '1955-03-04'],
    ['patient534', 'hasSmokingHistory', 'SmokingHistory'],
    ['patient534', 'hasBiomarker', 'ROS1'],    
    ['patient534', 'hasBiomarker', 'ALK'], 
    ['patient534', 'hasComorbidity', 'Dyslipidemia'],  
    ['patient534', 'hasComorbidity', 'Cardiopathy'],     
    
    ['patient1234', 'hasSmokingHistory', 'SmokingHistory'],
    ['patient1234', 'hasBiomarker', 'ROS1'],    
    ['patient1234', 'hasComorbidity', 'Dyslipidemia'],    

    ['patient98', 'hasrelapse', 'relapse'],    
    ['patient98', 'hasSmokingHistory', 'SmokingHistory'],    
    ['patient98', 'hasComorbidity', 'Dyslipidemia'], 
    
    ['patient721', 'hasProgression', 'progression'],
    ['patient25', 'hasCardiacArrest', 'CardiacArrest'], 
    ['relapse', 'therapytype', 'PalliativeCare'],
    
])
test_triples = np.array([
    ['patient1234', 'hasrelapse', 'relapse']
])

explanation_graph = np.array([
    ['patient1234', 'hasrelapse', 'relapse'],
    ['patient1234', 'hasSmokingHistory', 'SmokingHistory'],
    ['patient1234', 'hasBiomarker', 'ROS1'],    
    ['patient1234', 'hasComorbidity', 'Dyslipidemia'],
    ['relapse', 'therapytype', 'PalliativeCare'],
    ['patient1234', 'resembles', 'patient98'],    
    ['patient1234', 'resembles', 'patient534'],   
    ['relapse', 'resembles', 'relapse'],
    ['patient98', 'hasrelapse', 'relapse'],
    ['patient534', 'hasrelapse', 'relapse']
])

sets = {'subjects': {'patient98', 'patient1234', 'patient534'}, 'objects': {'CardiacArrest', 'relapse', 'progression'}, 'predicates': {'hasrelapse'}}

examples = [ 
             ['patient98', 'hasrelapse', 'relapse'], 
             ['patient534', 'hasrelapse', 'relapse'], 
             ['patient1234', 'resembles', 'patient98'], 
             ['patient1234', 'resembles', 'patient534'],
             ['relapse', 'resembles', 'relapse']           
           ]

prototype = [    
    ('patient1234', 'hasrelapse', 'relapse'),
    ('patient1234', 'hasSmokingHistory', 'SmokingHistory'),
    ('patient1234', 'hasComorbidity', 'Dyslipidemia'),
    ('relapse', 'therapytype', 'PalliativeCare')
]

prototype_permissive = [    
    ('patient1234', 'hasrelapse', 'relapse'),
    ('patient1234', 'hasSmokingHistory', 'SmokingHistory'),
    ('patient1234', 'hasComorbidity', 'Dyslipidemia'),
    ('relapse', 'therapytype', 'PalliativeCare'),
    ('patient1234', 'hasBiomarker', 'ROS1'),
    ('patient534', 'hasrelapse', 'relapse'),
    ('patient98', 'hasrelapse', 'relapse')
]

prototype_permissive_weights = [    
    ('patient1234', 'hasrelapse', 'relapse', '2'),
    ('patient1234', 'hasSmokingHistory', 'SmokingHistory', '2'),
    ('patient1234', 'hasComorbidity', 'Dyslipidemia', '2'),
    ('relapse', 'therapytype', 'PalliativeCare', '2'),
    ('patient1234', 'hasBiomarker', 'ROS1', '1'),
    ('patient534', 'hasrelapse', 'relapse', '1'), 
    ('patient98', 'hasrelapse', 'relapse', '1')
]

@pytest.fixture(scope=SCOPE)
def data_fb():
    X = load_fb15k_237(return_mapper=True)
    yield X

@pytest.fixture(scope=SCOPE)
def data():
    X = {'train': train_triples, 'test':test_triples, 'valid':[]}
    yield X    
    
@pytest.fixture(params = ["strict", "permissive"], scope=SCOPE)
def strategy(request):
    return request.param

@pytest.fixture(params = [1], scope=SCOPE)
def hop(request):
    return request.param

@pytest.fixture(params = [True, False], scope=SCOPE)
def weights(request):
    return request.param

@pytest.fixture(params = ["NE", "N", pytest.param("E", marks=pytest.mark.skip(reason="Not implemented"))], scope=SCOPE)
def mode(request):
    return request.param

@pytest.fixture(params = ["so"], scope=SCOPE)
def sampling_mode(request):
    return request.param

@pytest.fixture(params = [None], scope=SCOPE)
def epsilon(request):
    return request.param

@pytest.fixture(params = ['cosine', 'euclidean', 'manhattan', 'chebyshev', 'minkowski', pytest.param(['wminkowski', 'seuclidean', 'mahalanobis'], marks=pytest.mark.skip(reason="Not implemented"))], scope=SCOPE)
def metric(request):
    return request.param

@pytest.fixture(params = [3], scope=SCOPE)
def m(request):
    return request.param

@pytest.fixture(scope=SCOPE)
def model(data):
    model = TransE()    
#    model.is_fitted = True
    yield model 

def predict(x, y):
    return 0.8

def predict_proba(x, y):
    return 0.8

def get_embeddings(self, entities, embedding_type='entity'):
    embs = []
    if isinstance(entities, str):
        entities = [entities]
    for ent in entities:
        embs.append(embeddings[ent])
    return np.asarray(embs)

@pytest.fixture(scope=SCOPE)
def explainer(data, model, mocker):   
    mocker.patch(
        'ampligraph.compat.TransE.predict',
        predict
    )
    mocker.patch(
        'ampligraph.compat.TransE.predict_proba',
        predict_proba
    )

    mocker.patch(
        'ampligraph.compat.TransE.get_embeddings',
        get_embeddings
    ) 
    explainer = ExamplE(data, model)
    yield explainer

@pytest.fixture(scope=SCOPE)
def target_triple(request, data):
    triple = data['test'][0]
    return triple

@pytest.mark.skip(reason="Needs adjustment for new API")
def test_get_neighbourhood(explainer, m, target_triple, metric):
    neighbours = explainer.get_neighbourhood(target_triple[0], m=m, metric=metric)
    assert neighbours == set(["patient98", "patient534", "patient1234"]), "Not yet implemented"

@pytest.mark.skip(reason="Needs adjustment for new API")
def test_neighbourhood_sampler(explainer, sampling_mode, m, epsilon, target_triple, metric):
    actual_sets = explainer.neighbourhood_sampler(target_triple, mode=sampling_mode, m=m, epsilon=epsilon, metric=metric)
    assert actual_sets == sets, "Actual sets are not equal to the expected ones: expected {}, got {}".format(sets, actual_sets)

def is_equal_sets_of_lists(actual, expected):
    """Helper function to check wether two lists of lists
       contain the same lists (If lists of triples contain the same triples).

       Parameters
       ----------
       actual: first list.
       expected: second list.

       Returns
       -------
       True/False: if lists contain same elements no matter in which order.
    """
    if len(actual) == len(expected):
        print("Lengths equal")
        for elem in actual:
            if elem in expected:
                continue
            else:
                return False
    else:
        print("Lengths not equal, {} and {}".format(len(actual), len(expected)))
        return False
    return True

@pytest.mark.skip(reason="Needs adjustment for new API")
def test_examples_filter(explainer, target_triple):
    actual_examples = explainer.examples_filter(sets, target_triple)
    assert is_equal_sets_of_lists(actual_examples, examples), "Actual examples are not equal to the expected examples, expected {}, got {}. ".format(examples, actual_examples)
        
@pytest.mark.skip(reason="Needs adjustment for new API")      
def test_prototype_aggregator(explainer, target_triple, mode, strategy, hop, weights):
    actual_prototype, _ = explainer.prototype_aggregator(examples, target_triple, mode, strategy, hop, weights)
    print(actual_prototype)
    if strategy == "strict":
        assert is_equal_sets_of_lists(actual_prototype, prototype),\
        "Actual prototype is not equal to the expected prototype, expected {}, got {}.".format(prototype, actual_prototype)
    elif not weights:
        assert is_equal_sets_of_lists(actual_prototype, prototype_permissive),\
        "Actual prototype is not equal to the expected prototype, expected {}, got {}.".format(prototype_permissive, actual_prototype)
    else:
        assert is_equal_sets_of_lists(actual_prototype, prototype_permissive_weights), "Actual prototype is not equal to the expected prototype, expected {}, got {}.".format(prototype_permissive_weights, actual_prototype)
       
   
@pytest.mark.skip(reason="Needs different objects for different elements, see explain...")   
def test_explanation_graph_assembler(explainer, target_triple):
    actual_explanation_graph = explainer.explanation_graph_assembler(examples, prototype, target_triple)
    assert is_equal_sets_of_lists(actual_explanation_graph, explanation_graph), "Actual explanation graph is not equal to the expected explanation graph, expected {}, got {}.".format(explanation_graph, actual_explanation_graph)

@pytest.mark.skip(reason="Needs adjustment for new API")
def test_predict_explain(explainer, target_triple, strategy, hop, m, epsilon, mode, sampling_mode, metric):
    if strategy == 'permissive':
        explanation_dict = explainer.predict_explain(target_triple, mode=mode, strategy=strategy, hop=hop, sampling_mode=sampling_mode, m=m, epsilon=epsilon, metric=metric)
        actual_explanation = explainer.explanation_graph
        assert is_equal_sets_of_lists(actual_explanation, explanation_graph), "Actual explanation not equal to the expected explanation, expected {}, got {}.".format(explanation_graph, actual_explanation)
