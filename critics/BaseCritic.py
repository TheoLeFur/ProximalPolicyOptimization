class BaseCritic(object):

    def __init__(self, **kwargs):
        super(BaseCritic, self).__init__()

    def update(self, on, ac, next_ob, rew, term):
        raise NotImplementedError