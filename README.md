# TsetlinMachine-Checkers
Master's thesis project, University of Agder, Spring 2020, Checkers game classification by the use of Tsetlin Machine.
Thesis title: The Application of the Tsetlin Machine inCheckers

The research proposed a Tsetlin Machine predictor and a Checkers player using the Tsetlin Machine as its board predictor.
This is unfortunately not the most up-to-date collection of the scripts used, but the solution itself is complete.

The solution relies on the Tsetlin Machine library: https://github.com/cair/pyTsetlinMachineParallel
With the following install command pip pyTsetlinMachineParallel
Only compatible with the operating system Linux.
A custom installation, along with installation instructions, is located under "Tsetlin/pyTsetlinMachineParallel/"; which had its memory leak corrected.

The structure mostly follows the various dataset compositions created. The Checkers player can be found here: "data/TreeSearch/StandardEnd/TreeSearch.py"

The script for training and testing the proposed Tsetlin Machine configuration is located under various datasets' folders by the name of "StandardTsetlinWeightedPositiveBoost.py". Some of these scripts must be modified in order to have them run with the best found hyper-parameters.

The prototype Checkers player using weighted predictions found here does not contain the correction which it got on JupyterLab; which unfortunately has went offline. Apart from a correction in the way it found its clauses, the prototype is still present in the same folder as the proposed Checkers player, with the name "TreeSearchWeightScore.py".
