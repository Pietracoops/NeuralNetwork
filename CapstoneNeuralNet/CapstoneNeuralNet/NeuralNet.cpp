#include "NeuralNet.h"
using namespace std;

NeuralNet::NeuralNet(vector<int> NetConfig)
{
	//Constructor
	InitializeNeuralNet(NetConfig);

	//Init Constants
	BatchSize		= 0;											
	CurrentBatch	= 0;										
	OutputLayer		= 0;										
	CorrectAnswers	= 0;										

}

NeuralNet::~NeuralNet()
{
	//Destructor

}

void NeuralNet::InitializeNeuralNet(vector<int> NetConfig)
{
	//Setting NeuralNet Structure
	NeuralNetConfig.resize(NetConfig.size());
	OutputLayer = NetConfig.size() - 1;

	for (unsigned int i = 0; i < NetConfig.size(); i++)
	{
		NeuralNetConfig[i].resize(NetConfig[i]);
	}

	//Initializing Nodes with Random Values (starting at 1 because input nodes do not need initializing)
	for (unsigned int i = 1; i < NeuralNetConfig.size(); i++)
	{
		for (unsigned int j = 0; j < NeuralNetConfig[i].size(); j++)
		{
			NeuralNetConfig[i][j].Avalue = GenerateRand();								//Init A value
			NeuralNetConfig[i][j].Bvalue = GenerateRand();								//Init Bias Value

			NeuralNetConfig[i][j].Wvalue.resize(NeuralNetConfig[i - 1].size());			//Init Weights Array
			for (unsigned k = 0; k < NeuralNetConfig[i][j].Wvalue.size(); k++)			//Init Weights value
			{
				NeuralNetConfig[i][j].Wvalue[k] = GenerateRand();
			}
		}
	}
}


void NeuralNet::TrainNeuralNet(vector< vector<double>> Inputs, vector< vector<double>> Outputs, double DataRatio)
{
	DataInputs	= Inputs;
	DataOutputs = Outputs;

	StartTimer();


	cout << "Normalizing Data" << endl;
	//Normalize inputs
	NormalizeData();
	
	cout << "Shuffling Data" << endl;
	//Shuffle data
	ShuffleData();

	cout << "Splitting Data" << endl;
	//Split into Training and Testing Data
	SplitData(DataRatio);

	cout << "Start Training" << endl;
	//Train
	Train();

	cout << "Testing Network" << endl;
	//Test
	Test();

	StopTimer();
}




void NeuralNet::ShuffleData()
{
	int r = 0;
	vector<double> tmp;
	
	//Outputs should be same size as inputs
	for (unsigned int i = 0; i < DataInputs.size(); ++i)
	{
		r = (int)(GenerateRand() * DataInputs.size());
		tmp = DataInputs[r];
		DataInputs[r] = DataInputs[i];
		DataInputs[i] = tmp;

		tmp = DataOutputs[r];
		DataOutputs[r] = DataOutputs[i];
		DataOutputs[i] = tmp;
	}

}

void NeuralNet::NormalizeData()
{
	vector<double> sum;
	vector<double> mean;
	vector<double> standardDev;
	vector<double> standardDevSum;
	
	//Resizing
	sum.resize(DataInputs[0].size());
	mean.resize(sum.size());
	standardDev.resize(sum.size());
	standardDevSum.resize(sum.size());

	//Find Sum
	for (unsigned int j = 0; j < DataInputs[0].size(); j++)
	{
		for (unsigned int i = 0; i < DataInputs.size(); i++)			//First loop for sums
		{
			sum[j] += DataInputs[i][j];
		}
		
	}
	
	//Find Mean
	for (unsigned int i = 0; i < sum.size(); i++)
	{
		mean[i] = sum[i] / DataInputs.size();
	}
	//Find Standard Deviation Sum
	for (unsigned int i = 0; i < DataInputs.size(); i++)			//Second loop to normalize
	{
		for (unsigned int j = 0; j < DataInputs[i].size(); j++)
		{
			standardDevSum[j] += (DataInputs[i][j] - mean[j]) * (DataInputs[i][j] - mean[j]);
		}
	}
	//Find Standard Deviation
	for (unsigned int i = 0; i < sum.size(); i++)
	{
		standardDev[i] = sqrt(standardDevSum[i] / DataInputs.size());
	}
	//Normalize
	for (unsigned int i = 0; i < DataInputs.size(); i++)			//Second loop to normalize
	{
		for (unsigned int j = 0; j < DataInputs[i].size(); j++)
		{
			DataInputs[i][j] = (DataInputs[i][j] - mean[j]) * (1 / standardDev[j]);
		}
	}

}

void NeuralNet::SplitData(double DataRatio)
{

	//Initializing
	TrainingDataInputs.resize((int)(DataInputs.size() * DataRatio));
	TrainingDataOutputs.resize((int)(DataInputs.size() * DataRatio));
	TestDataInputs.resize(DataInputs.size() - TrainingDataInputs.size());
	TestDataOutputs.resize(DataInputs.size() - TrainingDataInputs.size());

	for (unsigned int i = 0; i < TrainingDataInputs.size(); i++)
	{
		TrainingDataInputs[i].resize(DataInputs[i].size());
		TrainingDataOutputs[i].resize(DataOutputs[i].size());
	}
	for (unsigned int i = 0; i < TestDataInputs.size(); i++)
	{
		TestDataInputs[i].resize(DataInputs[i].size());
		TestDataOutputs[i].resize(DataOutputs[i].size());
	}


	//Splitting Data
	for (unsigned int i = 0; i < TrainingDataInputs.size(); i++)		//Training Data
	{
		TrainingDataInputs[i] = DataInputs[i];
		TrainingDataOutputs[i] = DataOutputs[i];
	}
	for (unsigned int i = 0; i < DataInputs.size() - TrainingDataInputs.size(); i++)
	{
		TestDataInputs[i] = DataInputs[TrainingDataInputs.size() + i];
		TestDataOutputs[i] = DataOutputs[TrainingDataInputs.size() + i];

	}


}

void NeuralNet::Train()
{
	//Batch size - Stochastic Gradient descent, split up training sets into batches to calculate gradients
	unsigned int BatchCounter = 0;
	vector <Node> NeuralNetOutput;


	//Training Data
	for (unsigned int i = 0; i < TrainingDataInputs.size(); i++)
	{
		//Clear Vector
		NeuralNetOutput.clear();

		//Run Neural Net with Training data as input
		NeuralNetOutput = RunNeuralNet(TrainingDataInputs[i]);

		//Calculate Cost and store it
		StoreCost(NeuralNetOutput, TrainingDataOutputs[i]);
		
		//Calculate Gradient and store in Node
		CalculateGradient();

		BatchCounter++;
		CurrentBatch++;
		if (BatchCounter == BatchSize)
		{
			//Update Nodes
			UpdateNodes();
			//Restart BatchCounter
			BatchCounter = 0;
		}
		cout << "Training Data Completed: " << i << " / " << TrainingDataInputs.size() << endl;
	}


}

void NeuralNet::Test()
{
	vector <Node> NeuralNetOutput;
	CorrectAnswers = 0;
	//Test Data
	for (unsigned int i = 0; i < TestDataInputs.size(); i++)
	{

		//Clear Vector
		NeuralNetOutput.clear();

		//Calculate Testing Output
		NeuralNetOutput = RunNeuralNet(TestDataInputs[i]);

		//Check if Neural Net answer is correct
		CheckResults(NeuralNetOutput, i);

		cout << "Test Data Completed: " << i << " / " << TestDataInputs.size() << endl;
	}

	//Set Accuracy
	UpdateAccuracy();


}

vector<Node> NeuralNet::RunNeuralNet(vector<double> Inputs)
{

	//Initializing inputs
	for (unsigned int i = 0; i < Inputs.size(); i++)
	{
		NeuralNetConfig[0][i].Avalue = Inputs[i];
	}

	double Z = 0;
	for (unsigned int i = 1; i < NeuralNetConfig.size(); i++) //starting at 1 because we don't need to calculate inputs
	{
		for (unsigned int j = 0; j < NeuralNetConfig[i].size(); j++)
		{

			for (unsigned int k = 0; k < NeuralNetConfig[i - 1].size(); k++)
			{
				Z += NeuralNetConfig[i - 1][k].Avalue * NeuralNetConfig[i][j].Wvalue[k];
			}
			Z += NeuralNetConfig[i][j].Bvalue;
			NeuralNetConfig[i][j].Avalue = HyperTanFunction(Z);
		}

	}

	return NeuralNetConfig[NeuralNetConfig.size()];
}

void NeuralNet::StoreCost(vector<Node> Outputs, vector<double> ExpectedOutputs)
{
	double cost = 0;
	vector<double> temporaryOutputs;
	for (unsigned int i = 0; i < Outputs.size(); i++)
	{
		temporaryOutputs.push_back(Outputs[i].Avalue);

	}

	for (unsigned int i = 0; i < ExpectedOutputs.size(); i++)
	{
		cost += pow(temporaryOutputs[i] - ExpectedOutputs[i], 2);
	}

	Cost.push_back(cost);
}

void NeuralNet::CalculateGradient()
{

	vector<double> tempArray;

	for (unsigned int i = NeuralNetConfig.size(); i > 1; i--) //Ending at 1 because we don't need to calculate inputs
	{
		for (unsigned int j = 0; j < NeuralNetConfig[i].size(); j++)
		{
			tempArray.clear();
			tempArray.resize(NeuralNetConfig[i][j].Wvalue.size());
			for (unsigned int k = 0; k < NeuralNetConfig[i][j].Wvalue.size(); k++)
			{
				tempArray[k] = 2 * (1 / NeuralNetConfig[OutputLayer].size()) * (NeuralNetConfig[i][j].Avalue - DataOutputs[CurrentBatch][j]) * NeuralNetConfig[i][j].Avalue * (1 - NeuralNetConfig[i][j].Avalue) * NeuralNetConfig[i-1][k].Avalue;
			}
			NeuralNetConfig[i][j].WvalueDelta.push_back(tempArray);
			NeuralNetConfig[i][j].BvalueDelta.push_back(NeuralNetConfig[i][j].Avalue * (1 - NeuralNetConfig[i][j].Avalue) * 2 * (1 / NeuralNetConfig[OutputLayer].size()) * (NeuralNetConfig[i][j].Avalue - DataOutputs[CurrentBatch][j]));

		}
	}
}

void NeuralNet::UpdateNodes()
{
	for (unsigned int i = 1; i < NeuralNetConfig.size(); i++)
	{
		for (unsigned int j = 0; j < NeuralNetConfig[i].size(); j++)
		{
			NeuralNetConfig[i][j].UpdateWeights();
			NeuralNetConfig[i][j].UpdateBiases();
		}
	}

}

void NeuralNet::CheckResults(vector<Node> NeuralNetOutputs, int BatchCounter)
{
	int LargestIndex = 0;
	double LargestA = 0;
	for (unsigned int i = 0; i < NeuralNetOutputs.size(); i++)
	{
		if (NeuralNetOutputs[i].Avalue > LargestA)
		{
			LargestA = NeuralNetOutputs[i].Avalue;
			LargestIndex = i;
		}
	}


	int LargestTestIndex = 0;
	double LargestTestA = 0;
	for (unsigned int i = 0; i < TestDataOutputs[BatchCounter].size(); i++)
	{
		if (TestDataOutputs[BatchCounter][i] > LargestTestIndex)
		{
			LargestTestA = NeuralNetOutputs[i].Avalue;
			LargestTestIndex = i;
		}
	}

	if (LargestIndex == LargestTestIndex) CorrectAnswers++;

}

void NeuralNet::UpdateAccuracy()
{

	if (CorrectAnswers != 0 && TestDataInputs.size() != 0)
	{
		Accuracy = CorrectAnswers / TestDataInputs.size();
	}
}


//====================================================================================>
//SQUASHING FUNCTIONS
//====================================================================================>

double NeuralNet::HyperTanFunction(double x)
{
	if (x < -20.0) return -1.0; //approximation is correct to 30 decimals
	else if (x > 20.0) return 1.0;
	else return tanh(x);
}

double NeuralNet::HyperTanFunctionPrime(double x)
{
	//Derivative of tanh
	double tanhprime = 1 - (tanh(x)*tanh(x));
	return tanhprime;
}

double NeuralNet::Sigmoid(double x)
{
	double sig = 1 / (1 + exp(-x));

	return sig;
}

double NeuralNet::SigmoidPrime(double x)
{
	//Derivative of Sigmoid
	double sig = Sigmoid(x) * (1 - Sigmoid(x));

	return sig;
}


//====================================================================================>
//MISC FUNCTIONS
//====================================================================================>

void NeuralNet::StartTimer()
{
	StartTime = get_time_sec();
}

void NeuralNet::StopTimer()
{
	StopTime = get_time_sec();
}

double NeuralNet::ReturnTime()
{
	return StopTime - StartTime;
}