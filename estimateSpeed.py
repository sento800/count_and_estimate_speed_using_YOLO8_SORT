import numpy as np
import math


def estimateSpeed(location1, location2):
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	# ppm = location2[2] / carWidht
	ppm = 9
	d_meters = d_pixels / ppm
	print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
	fps = 15
	time_elapsed = 1 / fps * 3.6
	speed = d_meters * time_elapsed
	return speed