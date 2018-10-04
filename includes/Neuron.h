#ifndef NEURON_H
# define NEURON_H

#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

struct Connexion
{
	double weight;
	double deltaWeight;
};

class	Neuron
{
public:
	Neuron(unsigned int outputCount, unsigned int index);

	void feedForward(const Layer &prevLayer);
	void calcOutputGradient(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

	inline void setOutput(double v) { m_output = v; };
	inline double getOutput(void) const { return (m_output); };

private:
	double m_output;
	double m_gradient;
	unsigned int m_index;
	vector<Connexion> m_outputWeights;

	double sumDOW(const Layer &layer) const;

	static double eta;		//[0.0 <=> 1.0] taux d'apprentissage général
	static double alpha;	//[0.0 <=> n] facteur d'inertie de l'appretissage

	static double activationFunction(double d);
	static double activationFunctionDerivative(double d);
	static double randomWeight(void) { return (rand() / double(RAND_MAX)); }
};

#endif
