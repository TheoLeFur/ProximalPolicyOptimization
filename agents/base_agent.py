class BaseAgent(object):

    """
    An abstract class for a Reinforcement Learning Agent.
    """

    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    def train(self) -> dict:
        raise NotImplementedError
    
    def add_to_replay_buffer(self, paths):
        raise NotImplementedError
    
    
