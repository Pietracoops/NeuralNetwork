#pragma once
#include "Mass_ToolKit.h"
#include "Node.h"

class NeuralNet
{

private:

	//VARIABLES
	std::vector< std::vector<Node>>	NeuralNetConfig;
	std::vector< std::vector<double>> DataInputs;
	std::vector< std::vector<double>> DataOutputs;

	std::vector< std::vector<double>> TrainingDataInputs;
	std::vector< std::vector<double>> TrainingDataOutputs;
	std::vector< std::vector<double>> TestDataInputs;
	std::vector< std::vector<double>> TestDataOutputs;

	std::vector<double> Cost;								//Cost functions

	int BatchSize;											//Incremements in which nodes should be updated
	int CurrentBatch;										//Keeping track of Neural Net Epochs during training
	int OutputLayer;										//Used to reference the Output Layer of Neural Net
	int CorrectAnswers;										//Used to calculate Accuracy
	
	double StartTime;										//Start time for Neural Net Training
	double StopTime;										//End time for Neural Net Training

	void InitializeNeuralNet(std::vector<int> NetConfig);

	void ShuffleData();
	void NormalizeData();
	void SplitData(double DataRatio);
	void Train();
	void Test();
	void CheckResults(std::vector<Node> NeuralNetOutputs, int BatchCounter);
	void UpdateAccuracy();
	void StartTimer();
	void StopTimer();


	//Backpropogation
	void StoreCost(std::vector<Node> Outputs, std::vector<double> ExpectedOutputs);				//Store Cost into vectors
	void CalculateGradient();																	//Calculate Gradient
	void UpdateNodes();																			//Update Nodes based on Gradient
	
	double Sigmoid(double x);
	double SigmoidPrime(double x);
	double HyperTanFunction(double x);
	double HyperTanFunctionPrime(double x);

public:


	double Accuracy;


	NeuralNet(std::vector<int> NetConfig);  //Constructor
	~NeuralNet();

	void TrainNeuralNet(std::vector< std::vector<double>> Inputs, std::vector< std::vector<double>> Outputs, double DataRatio);
	double ReturnTime();

	std::vector<Node> RunNeuralNet(std::vector<double> Inputs);
};


