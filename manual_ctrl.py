from pynput.keyboard import Key, Listener, KeyCode


class ManualController:

    def __init__(self, commander):  # plotter
        self.cmd = commander
        # self.plotter = plotter

    def on_press(self, key):

        if key == KeyCode.from_char("w"):
            self.cmd.action('forward')
        elif key == KeyCode.from_char("s"):
            self.cmd.action('backward')
        elif key == KeyCode.from_char("a"):
            self.cmd.action('left')
        elif key == KeyCode.from_char("d"):
            self.cmd.action('right')

        if key == KeyCode.from_char("p"):
            self.cmd.sim_command('get_position')

        if key == Key.space:
            self.cmd.sim_command('save_view')

        if key == Key.tab:
            self.cmd.sim_command('change_view')

        if key == KeyCode.from_char("o"):
            self.cmd.get_observation(show=True)

        if key == KeyCode.from_char("g"):
            self.cmd.get_observation(grayscale=True, show=True)

        if key == KeyCode.from_char("r"):
            self.cmd.new_episode()

        # self.plotter.render()

    def on_release(self, key):
        if key == Key.esc:
            # Stop listener
            self.cmd.terminate_sim()
            return False
