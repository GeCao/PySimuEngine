import os



class AnimationActor:
    def __init__(self, resource_component):
        self.resource_component = resource_component

        self.mesh = None
        self.all_data = None

        self.simulator = None

        self.is_precomputed = True
        self.max_itr = None
        self.curr_itr = 0

        self.VAO = None
        self.VBO = None

        self.initialized = False

    def initialization(self, mesh=None, all_data=None, simulator=None, source_from='precomputed'):
        if source_from == 'precomputed':
            self.is_precomputed = True
            self.mesh = mesh
            self.all_data = all_data
            self.max_itr = all_data.shape[0]
        else:
            self.is_precomputed = False
            self.simulator = simulator
            self.mesh = mesh

        self.initialized = True

    def get_curr_itr(self):
        return self.curr_itr

    def update(self, data=None):
        if self.is_precomputed:
            self.mesh.update(data=self.all_data[self.curr_itr])
            self.curr_itr = (self.curr_itr + 1) % self.max_itr
        elif data is not None:
            self.mesh.update(data=data)
            self.curr_itr = (self.curr_itr + 1)
        else:
            # Not precomputed so we can not read from animation path,
            # No input data if this is a simulator animation, so just give it a pass
            return
