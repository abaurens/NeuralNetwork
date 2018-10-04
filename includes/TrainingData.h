#ifndef TRAINING_DATA_H
# define TRAINING_DATA_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

class	TrainingData
{
public:
	TrainingData(const string path);

	inline bool isEof() { return (m_trainingDatafile.eof()); };

	void getTopology(vector<unsigned int> &topology);

	unsigned int getNextInputs(vector<double> &inputVals);
	unsigned int getTargetOutputs(vector<double> &targetVals);

private:
	ifstream m_trainingDatafile;
};

#endif
