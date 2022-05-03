# This section of the project worked on by: Andrew Bruneel
# This is just a quick code segment used to ensure that google Colab did not
# disconnect when the model was ran overnight on GPU.

from pynput.mouse import Controller, Button
import time

mouse = Controller()

while True:
    mouse.click(Button.left, 1)
    print('clicked')

    time.sleep(5)
