"""Decoder that gets nodes labels from a mapper."""
import json
import numbers
import numpy as np


class Decoder:
    """Decoder - mapper for human readable labels (to be used when mapping
       dictionary for entities is available).

       Provides converting capability for
         - triples with decoder.convert(triples),
         - explanation dictionary returned by explainers (e.g. ExamplE, RandomExplainer).

       Example
       -------
       >>> decoder = Decoder('./human_labels_map.json')
       >>> explanation = {'query_triple': array(['a','b','c']), ...}
       >>> decoder.convert_dict(explanation)
       {'query_triple': array(['a_label', 'b', 'c_label'])
    """
    def __init__(self, mapper):
        """Decoder initializer.

           Parameters
           ----------
           mapper: str
               name of json file containing map between entities and their labels.
        """
        with open(mapper, 'r') as f:
            self.mapper = json.loads(f.read())

    def convert(self, triples):
        """Helper function that converts triple entities using provided mapper.

           Parameters
           ----------
           triples: np.array
               triples to be converted.

           Returns
           -------
           converted: np.array
               translated triples.

           Example
           -------
           >>> decoder = Decoder('./human_labels_map.json')
           >>> triples = array([['a','b','c'],...])
           >>> decoder.convert(triples)
           array([['a_label', 'b', 'c_label'],...])
        """
        trips = []
        for triple in triples:
            trips.append((self.mapper[triple[0]], triple[1], self.mapper[triple[2]]))
        converted = np.asarray(trips)
        return converted

    def convert_dict(self, expl):
        """Helper function to convert dictionary of explanation
           into human readable format provided by mapper.

           Parameters
           ----------
           expl: dict
               explanation dictionary as returned by explainers.

           Returns
           -------
           new_dict: dict
               explanation dictionary with translated labels.

           Example
           -------
           >>> decoder = Decoder('./human_labels_map.json')
           >>> explanation = {'query_triple': array(['a','b','c']), ...}
           >>> decoder.convert_dict(explanation)
           {'query_triple': array(['a_label', 'b', 'c_label'])

        """
        new_dict = {}
        for k in expl:
            if k == 'prediction' or k == 'probability' or isinstance(expl[k], numbers.Number):
                new_dict[k] = expl[k]
                continue
            if len(expl[k]) == 3 and isinstance(expl[k][0], str):
                tmp = [expl[k]]
            else:
                tmp = expl[k]
            new_dict[k] = self.convert(tmp)

        new_dict['query_triple'] = new_dict['query_triple'][0]
        return new_dict
