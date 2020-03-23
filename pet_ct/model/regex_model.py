import re
from pet_ct.data.labeler import Labeler
from pet_ct.data.report_transforms import extract_impression

class BaselineRegexModel():

    def __init__(self, task_configs, term_graph_path):
        self.task_to_config = {}
        for task_config in task_configs:
            self.task_to_config[task_config['task']] = task_config

        self.labeler = Labeler(term_graph_class='TermGraph',
                               term_graph_args={'graph_dir':
                                                'data/pet_ct_terms/main_regrouped/'})

    def predict(self, text):
        text = text.replace('ABDOMEN AND PELVIS:', '')
        text = text.replace('THORAX:', '')
        text = text.replace('HEAD AND NECK:', '')
        text = text.replace('MUSCULOSKELETAL:', '')
        text = text.lower()
        preds = {}
        for task, config in self.task_to_config.items():
            preds[task] = self.labeler.contains_term(text,
                                                     names=[task], hit_code=1,
                                                     miss_code=0, aggregate=True,
                                                     neg_regexes=["no", "not", "without", "physiologic", "unremarkable", "resolution"])[0][0]
        return preds

