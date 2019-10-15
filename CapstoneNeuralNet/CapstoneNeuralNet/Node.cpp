#include "NeuralNet.h"
using namespace std;


Node::Node()
{

	

}

Node::~Node()
{


}


void Node::UpdateWeights()
{
	double mean = 0;
	for (unsigned int i = 0; i < WvalueDelta[0].size(); i++)
	{
		for (unsigned int j = 0; j < WvalueDelta.size(); j++)
		{
			mean += WvalueDelta[j][i];
		}
		mean = mean / WvalueDelta.size();
		Wvalue[i] += mean;
	}
}

void Node::UpdateBiases()
{
	double mean = 0;
	for (int unsigned i = 0; i < BvalueDelta.size(); i++)
	{
		mean += BvalueDelta[i];
	}
	mean = 1 / BvalueDelta.size();
	Bvalue += mean;
}