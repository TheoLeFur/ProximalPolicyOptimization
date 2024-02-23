class BaseCritic(object):

    def __init__(self, **kwargs):
        super(BaseCritic, self).__init__()

    def update(self, states, **kwargs):
        raise NotImplementedError