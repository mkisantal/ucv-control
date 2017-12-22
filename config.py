
class Configuration:
    def __init__(self, mode, steps):

        # mode selector: training or evaluation
        if mode == 'train':
            self.TRAIN_MODE = True
        elif mode == 'eval':
            self.TRAIN_MODE = False
        else:
            raise ValueError('Mode has to be either \'train\' or \'eval\'')

        self.MAX_STEPS = int(steps)  # absolute episode count, for training
        self.MODEL_NAME = None

        # ---------------------------------------------------------------
        # SET THE PARAMETERS BELOW MANUALLY
        # Training Settings
        self.LOAD_MODEL = True  # if there is a saved model already
        self.MODEL_PATH = './model'
        self.MODEL_SAVE_PERIOD = 1e4
        self.LOGGING_PERIOD = 500
        self.VERBOSITY = 2

        # DRONE COMMANDS
        self.DRONE_SPEED = 60  # cm/step
        self.DRONE_ANGULAR_ACC = 5  # deg/step^2
        self.DRONE_MAX_ANGULAR_SPEED = 15  # deg/step

        # Basic RL settings
        self.MAX_EPISODE_LENGTH = 120
        self.STEPS_FOR_UPDATE = 5
        self.LEARNING_RATE = ScheduledParameter(1e-4, 1e-6, 5e5, 1e6)
        self.ENTROPY = ScheduledParameter(-0.2, -0.01, 5e5, 1e6)
        self.GAMMA = 0.99
        self.LAMBDA = 0.96
        self.STATE_SHAPE = [84, 84, 3]  # RGB
        self.ACTIONS = 3
        self.NUM_WORKERS = 8
        self.ACCELERATION_ACTIONS = True

        # Additional inputs
        self.GOAL_ON = True
        self.PREV_REWARD_ON = False
        self.PREV_ACTION_ON = True

        # RL rewards
        self.GOAL_DIRECTION_REWARD = 2.0
        self.CRASH_REWARD = -10.0
        self.CONTROL_EFFORT_REWARD_MULTIPLIER = -0.75 / 100
        self.TURNING_REWARD = -0.05

        # Auxiliary tasks
        self.AUX_TASK_D2 = True
        if mode == 'eval':
            self.AUX_TASK_D2 = False    # no aux task(s) for evaluation

        # Evaluation settings
        self.NUM_EVAL_WORKERS = 4
        self.STOCHASTIC_POLICY_EVAL = True
        self.MAX_EVALUATION_EPISODE_LENGTH = 400
        self.MAX_EPISODES_FOR_EVAL = 12
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


class ScheduledParameter:
    # set interpolation in schedule.py
    def __init__(self,  start_value, end_value, start_step, end_step):
        self.START_STEP = start_step
        self.END_STEP = end_step
        self.START_VALUE = start_value
        self.END_VALUE = end_value


