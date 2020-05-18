# default arguments
OUTPUT_PATH = 'result.png'
ITERATIONS = 600
CONTENT_WEIGHT = 1e-3
STYLE_WEIGHT = 1e6
LEARNING_RATE = 1e-1
REPORT_INTERVAL = 10
IMSAVE_INTERVAL = 100
RANDOM_START = False
STANDARD_TRAIN = True
STYLE_RESIZE = 300
SIZE_STEPS = 1
CONTENT_START_SIZE = 300

# set net parameter
# STYLE_LAYER = [1, 2, 3, 4, 6, 7, 9]  # could add more,maximal to 12
STYLE_LAYER = [1, 2, 3, 4, 5, 7, 8]  # could add more,maximal to 12
STYLE_LAYER_WEIGHTS = [15, 15, 1, 5, 5, 15,
                       15]  # this should be same length as STYLE_LAYER
# STYLE_LAYER_WEIGHTS = [21, 21, 1, 1, 1, 7,
#                        7]  # this should be same length as STYLE_LAYER

CONTENT_LAYER = [1, 2, 3]