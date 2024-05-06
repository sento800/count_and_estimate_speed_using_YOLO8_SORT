import math


def estimateSpeed(location1, location2,fps,w):
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	ppm = w / 1.7
	d_meters = d_pixels / ppm
	time_elapsed =  3.6 * fps
	speed = d_meters * time_elapsed
	return speed