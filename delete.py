import imageio as iio
import time

import matplotlib.pyplot as plt


camera = iio.get_reader('<video0>', input_params=['-framerate', '30'])
delay = 1/ 5

i = 0
frame = camera.get_next_data()

while True:
    i += 1

    im = camera.get_next_data()
    #print(dir(frame))
    iio.imwrite(f'./frames/{i}.png', im, format='png')


camera.close()
