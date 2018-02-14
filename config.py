
class Config:

    # Training Settings
    LOAD_MODEL = True
    MODEL_PATH = './model'
    VERBOSITY = 1
    LOGGING_PERIOD = 500
    MODEL_SAVE_PERIOD = 1e4

    # Neural Network Settings
    USE_LSTM = False

    # Basic RL settings
    MAX_EPISODE_LENGTH = 30
    STEPS_FOR_UPDATE = 5
    GAMMA = 0.99
    STATE_SHAPE = [84, 84, 3]  # RGB
    ACTIONS = 3
    NUM_WORKERS = 8
    MAX_STEPS = 2e6
    GOAL_ON = True
    TERMINATE_AT_GOAL = True
    ACCELERATION_ACTIONS = True
    GOAL_DIRECTION_REWARD = 1.0
    CRASH_REWARD = -10.0
    TURN_REWARD = - 0.01

    # Auxiliary tasks
    AUX_TASK_D2 = True

    # Evaluation
    EVAL_MODE = False
    INTEGRATED_EVAL = True
    MAX_EVALUATION_EPISODE_LENGTH = 600
    EPISODES_FOR_EVAL = 10
    EVAL_GOAL_X = 4000
    EVAL_GOAL_Y = -4000
    EVAL_START_X = -4000
    EVAL_START_Y = 4000

    # Simulator settings
    HOST = 'localhost'
    PORT = 9000
    # SIM_DIR = '/home/mate/ucv-pkg-labyrinth/LinuxNoEditor/unrealCVfirst/Binaries/Linux/'
    # SIM_NAME = 'unrealCVfirst-Linux-Shipping'
    # SIM_DIR = '/home/mate/Documents/ucv-pkg-outdoor5/LinuxNoEditor/outdoor/Binaries/Linux/'
    # SIM_NAME = 'outdoor'


    SIM_DIR = '/home/mate/ucv-pkg-outdoor-8-lite/LinuxNoEditor/outdoor_lite/Binaries/Linux/'
    SIM_NAME = 'outdoor_lite'

    SIM_DIR_LIST = ['/home/mate/ucv-pkg-outdoor-8-lite/LinuxNoEditor/outdoor_lite/Binaries/Linux/',
                         '/home/mate/ucv-pkg-outdoor-8-lite1/LinuxNoEditor/outdoor_lite/Binaries/Linux/',
                         '/home/mate/ucv-pkg-outdoor-8-lite2/LinuxNoEditor/outdoor_lite/Binaries/Linux/',
                         '/home/mate/ucv-pkg-outdoor-8-lite3/LinuxNoEditor/outdoor_lite/Binaries/Linux/',
                         '/home/mate/ucv-pkg-outdoor-8-lite4/LinuxNoEditor/outdoor_lite/Binaries/Linux/',
                         '/home/mate/ucv-pkg-outdoor-8-lite5/LinuxNoEditor/outdoor_lite/Binaries/Linux/',
                         '/home/mate/ucv-pkg-outdoor-8-lite6/LinuxNoEditor/outdoor_lite/Binaries/Linux/',
                         '/home/mate/ucv-pkg-outdoor-8-lite7/LinuxNoEditor/outdoor_lite/Binaries/Linux/']

    RANDOM_SPAWN_LOCATIONS = True
    MAP_X_MIN = -4000
    MAP_X_MAX = 4000
    MAP_Y_MIN = -4000
    MAP_Y_MAX = 4000


