"""
All viable commands that can be sent to the engine
"""

NORTH = 'n'
SOUTH = 's'
EAST = 'e'
WEST = 'w'
STAY_STILL = 'o'
# command for shipyard to generate new ship
GENERATE = 'g'
# command for ship to become a dropoff spot
CONSTRUCT = 'c'
# command necessary for all move commands
# e.g. '<MOVE> <id> <direction>', '<MOVE> <id> <STAY_STILL>'
MOVE = 'm'
