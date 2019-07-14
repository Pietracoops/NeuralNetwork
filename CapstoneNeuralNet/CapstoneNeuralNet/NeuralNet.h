#pragma once

class NeuralNet
{

private:

	//VARIABLES


	double rnd;   //For initializeWeights() and Shuffle()

	int numInput;
	int numHidden;
	int numOutput;

	int numTestData;

	double* inputs;

	double** ihWeights; //input-hidden
	double* hBiases;
	double* hOutputs;

	double** hoWeights; //hidden-ouput
	double* oBiases;

	double* outputs;

	double* oGrads; //output gradients for back-propogation
	double* hGrads; //hidden gradients for back-propogation

	// back-prop momentum specific arrays (could be local to method Train)
	double** ihPrevWeightsDelta; //For momentum with back-propogation
	double* hPrevBiasesDelta;
	double** hoPrevWeightsDelta;
	double* oPrevBiasesDelta;



	//FUNCTIONS

	double** MakeMatrix(int rows, int cols);
	double GenerateRand();
	void Shuffle(int* Sequence);
	double MeanSquaredError(double** TrainData);
	void CopyArray(double* SourceArray, int SourceArrayIdx, double* DestinationArray, int DestinationIndex, int length);
	double HyperTanFunction(double x);
	double* Softmax(double* oSums);
	void UpdateWeights(double* tValues, double learnRate, double momentum, double weightDecay);

public:

	//VARIABLES



	//FUNCTIONS
	NeuralNet(int numInput, int numHidden, int numOutput);  //Constructor
	~NeuralNet();

	void InitializeWeights();
	void SetWeights(double* weights);
	void Train(double** trainData, int maxEpochs, double learnRate, double momentum, double weightDecay);
	double* ComputeOutputs(double* xValues);
};