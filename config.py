
class Configuration:
    def __init__(self, mode, episodes):

        # mode selector: training or evaluation
        if mode == 'train':
            self.TRAIN_MODE = True
        elif mode == 'eval':
            self.TRAIN_MODE = False
        else:
            raise ValueError('Mode has to be either \'train\' or \'eval\'')

        self.MAX_STEPS = int(episodes)  # absolute episode count, for training

        # ---------------------------------------------------------------
        # SET THE PARAMETERS BELOW MANUALLY
        # Training Settings
        self.LOAD_MODEL = False
        self.MODEL_PATH = './model'
        self.MODEL_SAVE_PERIOD = 5 * 10e4
        self.LOGGING_PERIOD = 500
        self.VERBOSITY = 1

        # Basic RL settings
        self.MAX_EPISODE_LENGTH = 120
        self.STEPS_FOR_UPDATE = 5
        self.GAMMA = 0.99
        self.STATE_SHAPE = [84, 84, 3]  # RGB
        self.ACTIONS = 3
        self.NUM_WORKERS = 8
        self.GOAL_ON = True
        self.ACCELERATION_ACTIONS = True

        # RL rewards
        self.GOAL_DIRECTION_REWARD = 1.0
        self.CRASH_REWARD = -10.0
        self.CONTROL_EFFORT_REWARD_MULTIPLIER = -1.0 / 100

        # Auxiliary tasks
        self.AUX_TASK_D2 = True
        if mode == 'eval':
            self.AUX_TASK_D2 = False    # no aux task(s) for evaluation

        # Evaluation settings
        self.MAX_EVALUATION_EPISODE_LENGTH = 750
        self.MAX_EPISODES_FOR_EVAL = 20
        self.EVAL_GOAL_X = 4000
        self.EVAL_GOAL_Y = -4000
        self.EVAL_START_X = -4000
        self.EVAL_START_Y = 4000

        # Simulator settings
        self.HOST = 'localhost'
        self.PORT = 9000

        self.SIM_DIR = '/home/mate/ucv-pkg-outdoor-8-lite/LinuxNoEditor/outdoor_lite/Binaries/Linux/'
        self.SIM_NAME = 'outdoor_lite'

        self.SIM_DIR_LIST = ['/home/mate/ucv-pkg-outdoor-8-lite/LinuxNoEditor/outdoor_lite/Binaries/Linux/',
                             '/home/mate/ucv-pkg-outdoor-8-lite/LinuxNoEditor/outdoor_lite/Binaries/Linux/']

        self.RANDOM_SPAWN_LOCATIONS = True
        self.MAP_X_MIN = -4000
        self.MAP_X_MAX = 4000
        self.MAP_Y_MIN = -4000
        self.MAP_Y_MAX = 4000


