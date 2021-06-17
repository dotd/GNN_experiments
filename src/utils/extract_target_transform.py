class ExtractTargetTransform(object):
    """
    A transform that chooses the specified target labels for a sample
    For example, it is useful for datasets with multiple tasks if we want to train a model on a single task
    """
    def __init__(self, target):
        self.target = target

    def __call__(self, data):
        data.y = data.y[:, self.target]
        return data
