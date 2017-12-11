import tensorflow.python as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope


def get_variables_to_restore(scope=None, suffix=None, collection=ops.GraphKeys.VARIABLES):

    if isinstance(scope, variable_scope.VariableScope):
        scope = scope.name
    if suffix is not None:
        if ':' not in suffix:
            suffix += ':'
        scope = (scope or '') + '.*' + suffix
    return ops.get_collection(collection, scope)
