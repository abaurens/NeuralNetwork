#include <vector>
#include <cassert>

#include "NeuralNetwork.h"
#include "TrainingData.h"
#include "Network.h"

void showVectorVals(string label, std::vector<double> &v)
{
	std::cout << label << " ";
	for (unsigned int i = 0; i < v.size(); ++i)
		std::cout << v[i] << " ";
	std::cout << std::endl;
}

int main()
{
	TrainingData trainData("./test.txt");

	std::vector<unsigned int> topology;

	trainData.getTopology(topology);
	Net network(topology);

	std::vector<double> inputVals;
	std::vector<double> targetVals;
	std::vector<double> resultVals;
	unsigned int trainingPass = 0;

	while (!trainData.isEof())
	{
		++trainingPass;
		std::cout << std::endl << "Pass " << trainingPass;

		//Acquisition d'une nouvelle donnée d'entrainement
		if (trainData.getNextInputs(inputVals) != topology[0])
			break;
		showVectorVals(": Inputs:", inputVals);
		network.feedForward(inputVals);

		//Collecte des résultats actuels du réseau
		network.getResult(resultVals);
		showVectorVals(": Outputs:", resultVals);

		//Apprend au réseau quelle résultat était attendu
		trainData.getTargetOutputs(targetVals);
		showVectorVals(": Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		network.backProp(targetVals);

		//Rapport sur l'avancement de l'entrainement
		std::cout << "Net recent average error: " << network.getRecentAverageError() << std::endl;
	}

	std::cout << std::endl << "Done !" << std::endl;

	return (0);
}
