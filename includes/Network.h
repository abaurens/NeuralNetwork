#ifndef NETWORK_H
# define NETWORK_H

using namespace std;


# include <vector>
# include <cassert>
# include <iostream>
# include "Neuron.h"

class	Net
{
public:
	Net(const vector<unsigned int> &topology);

	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResult(vector<double> &resultVals) const;

	inline double getRecentAverageError(void) const { return (m_recentAverageError); };

private:
	vector<Layer> m_layers;
	double m_error;

	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};

#endif
