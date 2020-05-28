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

