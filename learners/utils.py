from typing import Union, Type

from learners.multi_label_learner import MultiLabelLearner
from learners.single_label_learner import SingleLabelLearner


def get_learner(is_multilabel: bool) -> Type[Union[MultiLabelLearner, SingleLabelLearner]]:
    if is_multilabel:
        return MultiLabelLearner
    else:
        return SingleLabelLearner
