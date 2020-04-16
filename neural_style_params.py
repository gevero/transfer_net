# default arguments
OUTPUT_PATH = 'result.jpg'
ITERATIONS = 800
CONTENT_WEIGHT = 1e-3
STYLE_WEIGHT = 1
LEARNING_RATE = 0.005
REPORT_INTERVAL = 5
IMSAVE_INTERVAL = 5
RANDOM_START = False
STANDARD_TRAIN = False
STYLE_RESIZE = None

# set net parameter
STYLE_LAYER = [1, 2, 3, 4, 6, 7, 9]  # could add more,maximal to 12
STYLE_LAYER_WEIGHTS = [21, 21, 1, 1, 1, 7,
                       7]  # this should be same length as STYLE_LAYER
CONTENT_LAYER = [1, 2, 3]