
class Config:

    # Training Settings
    LOAD_MODEL = False
    MODEL_PATH = './model'
    MODEL_SAVE_FREQ = 250
    VERBOSITY = 1

    # Basic RL settings
    MAX_EPISODE_LENGTH = 300
    GAMMA = 0.99
    STATE_SHAPE = [84, 84, 3]  # RGB
    ACTIONS = 3
    NUM_WORKERS = 1
    MAX_EPISODES = 10
    GOAL_ON = True

    # Auxiliary tasks
    AUX_TASK_D2 = True

    # Simulator settings
    HOST = 'localhost'
    PORT = 9000
    SIM_DIR = '/home/mate/Documents/ucv-pkg3/LinuxNoEditor/unrealCVfirst/Binaries/Linux/'
    SIM_NAME = 'unrealCVfirst-Linux-Shipping'

