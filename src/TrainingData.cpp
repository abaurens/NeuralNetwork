#include "TrainingData.h"

TrainingData::TrainingData(const string path)
{
	m_trainingDatafile.open(path.c_str());
}

void TrainingData::getTopology(vector<unsigned int> &topology)
{
	string line;
	string label;

	getline(m_trainingDatafile, line);
	stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology:") != 0)
	{
		abort();
	}

	while (!ss.eof())
	{
		unsigned int n;

		ss >> n;
		topology.push_back(n);
	}

	return ;
}

unsigned int TrainingData::getNextInputs(vector<double> &inputVals)
{
	string line;
	string label;
	getline(m_trainingDatafile, line);
	stringstream ss(line);

	inputVals.clear();

	ss >> label;
	if (label.compare("in:") == 0)
	{
		double oneValue;
		while (ss >> oneValue)
			inputVals.push_back(oneValue);
	}
	return (inputVals.size());
}

unsigned int TrainingData::getTargetOutputs(vector<double> &targetVals)
{
	string line;
	string label;
	getline(m_trainingDatafile, line);
	stringstream ss(line);

	targetVals.clear();

	ss >> label;
	if (label.compare("out:") == 0)
	{
		double oneValue;
		while (ss >> oneValue)
			targetVals.push_back(oneValue);
	}

	return (targetVals.size());
}
