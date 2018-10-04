#include "Neuron.h"

double Neuron::eta = 0.15;	//[0.0 <=> 1.0] taux d'apprentissage général
double Neuron::alpha = 0.5;	//[0.0 <=> n] facteur d'inertie de l'appretissage

Neuron::Neuron(unsigned int outputCount, unsigned int index)
{
	for (unsigned int i = 0; i < outputCount; ++i)
	{
		m_outputWeights.push_back(Connexion());
		m_outputWeights.back().weight = randomWeight();
	}
	m_index = index;
}

double Neuron::activationFunction(double d)
{
	return (tanh(d));
}

double Neuron::activationFunctionDerivative(double d)
{
	return (1.0 - (d * d));
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum;

	sum = 0.0;
	for (unsigned int neuron = 0; neuron < prevLayer.size(); ++neuron)
	{
		sum += prevLayer[neuron].getOutput() * prevLayer[neuron].m_outputWeights[m_index].weight;
	}

	m_output = Neuron::activationFunction(sum);
}

void Neuron::calcOutputGradient(double targetVal)
{
	double delta;

	delta = targetVal - m_output;
	m_gradient = delta * Neuron::activationFunctionDerivative(m_output);
}

double Neuron::sumDOW(const Layer &layer) const
{
	double sum;

	//somme de la contribution de cette couche à l'erreur globale
	sum = 0.0;
	for (unsigned int neuron = 0; neuron < layer.size() - 1; ++neuron)
		sum += m_outputWeights[neuron].weight * layer[neuron].m_gradient;
	return (sum);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::activationFunctionDerivative(m_output);
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
	double oldDeltaWeight;
	double newDeltaWeight;

	//les poids à mettre à jour sont dans la structure "Connexion"
	//des neurones de la couche précédente

	for (unsigned int n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		oldDeltaWeight = neuron.m_outputWeights[m_index].deltaWeight;

		newDeltaWeight = eta * neuron.getOutput() * m_gradient + alpha * oldDeltaWeight;

		neuron.m_outputWeights[m_index].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_index].weight += newDeltaWeight;
	}
}
