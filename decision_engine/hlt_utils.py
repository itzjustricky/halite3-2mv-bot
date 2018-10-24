"""
Additional utilities for getting information on Halite Game State.

"""

import typing

import numpy as np

from hlt.game_map import Player
from hlt.networking import Game
from hlt.positional import Position
import hlt.constants as hlt_constants

# TODO: cache this call, it only needs to be computed once
def get_enemy_player_ids(game: Game) -> typing.List[int]:
    """ Function to get the ids of all the enemy players

    :param game: object to represent the entire state of the game
    """
    all_player_ids = list(game.players.keys())
    all_player_ids.remove(game.my_id)

    return all_player_ids


def compute_ship_presence(
        player: Player,
        game_dimensions: typing.Tuple[int, int]) -> np.ndarray:
    """ Computes a player's ship positioning on the board. Returns a matrix with
        0/1 values; with value 1 indicating there is the passed player's ship
        on the coordinate and value 0 otherwise.

    :param player: player object storing state of a player as well API to
        execute commands for a player
    :param game_dimensions: tuple of (<height>, <width>) for the game board
    :returns: a numpy 2D Matrix of
    """
    ship_presence = np.zeros(game_dimensions)
    for ship in player.get_ships():
        x, y = ship.position.x, ship.position.y
        ship_presence[x][y] = 1
    return ship_presence


def compute_halite_map(game: Game) -> np.ndarray:
    """ Place the halite amount per coordinate in the Game Board
        into a numpy matrix
    """
    game_dimensions = game.height, game.width
    halite_map = np.empty(game_dimensions)

    row_inds, column_inds = range(game.height), range(game.width)
    for x, y in itertools.product(row_inds, column_inds):
        halite_map[x][y] = game[Position(x, y)].halite_amount
    return halite_map


def compute_inspiration_map(game: Game) -> np.ndarray:
    """
    Computes the locations on the Game board in which the ships can become inspired.

    :param game: TODO
    """
    game_dimensions = game.height, game.width
    inspiration_map = np.zeros(game_dimensions)

    if not hlt_constants.INSPIRATION_ENABLED:
        return inspiration_map

    enemy_player_ids = get_enemy_player_ids(game)
    # this is a matrix with values 0/1, where value 1 indicates there
    # is an enemy ship on the coordinate for any one of the enemies
    all_enemy_map = np.zeros(game_dimensions)
    for player_id in enemy_player_ids:
        all_enemy_map += compute_ship_presence(
            game.players[player_id], game_dimensions)

    # do some calculations
    inspiration_radius = hlt_constants.INSPIRATION_RADIUS
    needed_ship_count = hlt_constants.INSPIRATION_SHIP_COUNT

    # TODO: do some calculation to determine which points
    # our ships can be inspired at

    return inspiration_map
