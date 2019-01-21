class DQNConfig(object):

    def __init__(self, FLAGS):

        # ----------- Agent Params
        self.scale = 100
        self.display = False

        self.max_step = 5000 * self.scale
        self.memory_size = 100 * self.scale

        self.batch_size = 32
        self.random_start = 30
        self.cnn_format = 'NCHW'
        self.discount = 0.99 # epsilon in RL (decay index)
        self.target_q_update_step = 1 * self.scale
        self.learning_rate = 0.00025
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 5 * self.scale

        self.ep_end = 0.1
        self.ep_start = 1.
        self.ep_end_t = self.memory_size

        self.history_length = 4
        self.train_frequency = 4
        self.learn_start = 5. * self.scale

        self.min_delta = -1
        self.max_delta = 1

        self.double_q = False
        self.dueling = False

        self._test_step = 5 * self.scale
        self._save_step = self._test_step * 10

        # ----------- Environment Params
        self.env_name = 'Breakout-v0'

        # TODO: In furthur case you can just use local infomation with 64*64 or 32*32 pixel.
        self.screen_width  = 512
        self.screen_height = 424

        self.max_reward = 1.
        self.min_reward = -1.

        # ----------- Model Params
        self.model = 'm1'
        self.backend = 'tf'
        self.env_type = 'detail'
        self.action_repeat = 4
        self.ckpt_dir = r'../checkpoint'
        self.model_dir = r'../model'
        self.is_train = True

        if FLAGS['use_gpu'] == False:
            self.cnn_format = 'NHWC'
        else:
            self.cnn_format = 'NCHW'

        if FLAGS['is_train'] == False:
            self.is_train = False
            
    def list_all_member(self):
        params = {}
        for name,value in vars(self).items():
            params[name] = value
        return params
