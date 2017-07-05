from pynput.keyboard import Key, Listener, KeyCode


class ManualController:

    def __init__(self, commander):
        self.cmd = commander

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

    @staticmethod
    def on_release(key):
        if key == Key.esc:
            # Stop listener
            return False
