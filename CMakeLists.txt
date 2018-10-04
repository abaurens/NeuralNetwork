# CMakeList.txt : projet CMake pour NeuralNetwork, incluez la source et définissez
# la logique spécifique au projet ici.
#
cmake_minimum_required (VERSION 3.8)

set(SOURCES
	src/NeuralNetwork.cpp
	src/TrainingData.cpp
	src/Network.cpp
	src/Neuron.cpp
)

include_directories(includes/)

# Ajoutez une source à l'exécutable de ce projet.
add_executable (NeuralNetwork ${SOURCES})

# TODO: Ajoutez des tests et installez des cibles si nécessaire.
