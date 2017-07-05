import unrealcv
from manual_ctrl import ManualController
from pynput.keyboard import Listener
from command import Commander

# setup
ManualControlEnabled = True
(HOST, PORT) = ('localhost', 9000)
client = unrealcv.Client((HOST, PORT))

# connecting to UE4 UnrealCV server
client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')


cmd = Commander(client, goal_heading_deg=90)
manual = ManualController(cmd)


if ManualControlEnabled:
    with Listener(on_press=manual.on_press, on_release=manual.on_release) as listener:
        listener.join()
