#pragma once
#include "Mass_ToolKit.h"

class Node
{

public:

	//Node Values
	double Avalue;									//Node Value
	double Bvalue;									//Bias value
	std::vector<double> BvalueDelta;				//Bias Deltas
	std::vector<double> Wvalue;						//Weight values
	std::vector<std::vector<double>> WvalueDelta;	//Weight Deltas



	//FUNCTIONS
	Node();											//Constructor
	~Node();										//Destructor

	void UpdateWeights();							//Update Weight Values
	void UpdateBiases();							//Update Bias Values

};