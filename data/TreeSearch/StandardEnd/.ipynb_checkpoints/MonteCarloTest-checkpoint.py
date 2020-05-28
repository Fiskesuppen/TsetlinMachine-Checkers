
import importlib

modulename = "montecarlo.node"
importlib.import_module(modulename)
modulename = "montecarlo.montecarlo"
importlib.import_module(modulename)



#from chess import Game
#from montecarlo.node import Node
#from montecarlo.montecarlo import MonteCarlo

chess_game = Game()
montecarlo = MonteCarlo(Node(chess_game))