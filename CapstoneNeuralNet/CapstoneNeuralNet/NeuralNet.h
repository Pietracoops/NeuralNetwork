#pragma once
#include "Mass_ToolKit.h"
class NeuralNet
{

private:

	//VARIABLES


	double rnd;   //For initializeWeights() and Shuffle()

	int numInput;
	int numHidden;
	int numOutput;

	int numTestData;

	std::vector<double> inputs;

	std::vector< std::vector<double> > ihWeights;
	std::vector<double> hBiases;
	std::vector<double> hOutputs;

	std::vector< std::vector<double> > hoWeights;
	std::vector<double> oBiases;
	std::vector<double> outputs;

	std::vector<double> hGrads;
	std::vector<double> oGrads;

	std::vector< std::vector<double> > ihPrevWeightsDelta;
	std::vector<double> hPrevBiasesDelta;
	std::vector< std::vector<double> > hoPrevWeightsDelta;
	std::vector<double> oPrevBiasesDelta;



	//FUNCTIONS

	double** MakeMatrix(int rows, int cols);
	void Shuffle(std::vector<int>& Sequence);
	double MeanSquaredError(std::vector< std::vector<double> >& TrainData);
	void CopyArray(std::vector<double>& SourceArray, int SourceArrayIdx, std::vector<double>& DestinationArray, int DestinationIndex, int length);
	double HyperTanFunction(double x);
	std::vector<double> Softmax(std::vector<double> oSums);
	void UpdateWeights(std::vector<double> tValues, double learnRate, double momentum, double weightDecay);
	int MaxIndex(std::vector<double> vector);

	void StoreDatabase(int i);
public:

	//VARIABLES

	int numWeights;

	//FUNCTIONS
	NeuralNet(int numInput, int numHidden, int numOutput);  //Constructor
	~NeuralNet();


	void MakeTrainTest(std::vector< std::vector<double> >& AllData, std::vector< std::vector<double> >& TrainData, std::vector< std::vector<double> >& TestData);

	void InitializeWeights();
	void SetWeights(std::vector<double> weights);
	void GetWeights(std::vector<double>& result);
	void ExportWeights(std::string filename);
	void Train(std::vector< std::vector<double> >& trainData, int maxEpochs, double learnRate, double momentum, double weightDecay);
	std::vector<double> ComputeOutputs(std::vector<double> xValues);
	double Accuracy(std::vector< std::vector<double> >& testData);
	void Normalize(std::vector< std::vector<double> >& DataMatrix, std::vector<int> cols);
};