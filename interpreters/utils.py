from typing import Union, Type

from interpreters.multi_label_interpreter import MultiLabelInterpreter
from interpreters.single_label_interpreter import SingleLabelInterpreter


def get_interpreter(is_multilabel: bool) -> Type[Union[MultiLabelInterpreter, SingleLabelInterpreter]]:
    if is_multilabel:
        return MultiLabelInterpreter
    else:
        return SingleLabelInterpreter
