#include "network.h"

double Net::m_recentAverageSmoothingFactor = 100.0;

Net::Net(const vector<unsigned int> &topology)
{
	unsigned int layerCount;
	unsigned int outputCount;

	layerCount = topology.size();
	for (unsigned int layer = 0; layer < layerCount; ++layer)
	{
		m_layers.push_back(Layer());
		outputCount = topology[layer + 1];
		if (layer == topology.size() - 1)
			outputCount = 0;

		for (unsigned int neuron = 0; neuron <= topology[layer]; ++neuron)
		{
			m_layers.back().push_back(Neuron(outputCount, neuron));
			cout << "Made a Neuron!" << endl;
		}
		//Force les noeuds de biais à une valeur de 1.0 (le dernier noeud à être créé à chaque couche)
		m_layers.back().back().setOutput(1.0);
	}
}

void Net::feedForward(const vector<double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);

	//Assignation des valeurs de départ a la premiere couche de neurone
	for (unsigned int i = 0; i < inputVals.size(); ++i)
	{
		m_layers[0][i].setOutput(inputVals[i]);
	}

	//propagation de l'information entre les neurones
	for (unsigned int layer = 1; layer < m_layers.size(); ++layer)
	{
		Layer &prevLayer = m_layers[layer - 1];
		for (unsigned int neuron = 0; neuron < m_layers[layer].size() - 1; ++neuron)
		{
			m_layers[layer][neuron].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const vector<double> &targetVals)
{
	unsigned int neuronIdx;
	unsigned int layerIdx;
	double delta;

	//Calcul de l'erreur globale
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for (neuronIdx = 0; neuronIdx < outputLayer.size() - 1; ++neuronIdx)
	{
		delta = targetVals[neuronIdx] - outputLayer[neuronIdx].getOutput();
		m_error += delta * delta;
	}
	m_error = sqrt(m_error / outputLayer.size());

	// Implementation d'une mesure de performance
	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) /
		(m_recentAverageSmoothingFactor + 1.0);

	//Calcul du gradient de la couche de sortie
	for (neuronIdx = 0; neuronIdx < outputLayer.size() - 1; ++neuronIdx)
	{
		outputLayer[neuronIdx].calcOutputGradient(targetVals[neuronIdx]);
	}

	//Calcul du gradient des couches cachées

	for (unsigned int layerIdx = m_layers.size() - 2; layerIdx > 0; --layerIdx)
	{
		Layer &layer = m_layers[layerIdx];
		Layer &nextLayer = m_layers[layerIdx + 1];

		for (neuronIdx = 0; neuronIdx < layer.size(); ++neuronIdx)
		{
			layer[neuronIdx].calcHiddenGradients(nextLayer);
		}
	}

	//Pour chaque couche à partir de la couche de sortie jusqu'à la première couche cachée,
	// mettre à jour les poids de connexion

	for (layerIdx = m_layers.size() - 1; layerIdx > 0; --layerIdx)
	{
		Layer &layer = m_layers[layerIdx];
		Layer &prevLayer = m_layers[layerIdx - 1];

		for (neuronIdx = 0; neuronIdx < layer.size(); ++neuronIdx)
		{
			layer[neuronIdx].updateInputWeights(prevLayer);
		}
	}
}

void Net::getResult(vector<double> &resultVals) const
{
	resultVals.clear();

	for (unsigned int n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutput());
	}
}
