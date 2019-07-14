#include "NeuralNet.h"
#include <iostream>
#include <stdlib.h>		// srand, rand
#include <time.h>		// time
#include <math.h>
using namespace std;



//vector< vector<int> > vec;
//
//for (int i = 0; i < 10; i++) {
//	vector<int> row; // Create an empty row
//	for (int j = 0; j < 20; j++) {
//		row.push_back(i * j); // Add an element (column) to the row
//	}
//	vec.push_back(row); // Add the row to the main vector
//}

//vector<double> v(5);
//vector< vector<double> > m(3, vector<double>(3));
//
//m[2][2] = 4.4;
//
//cout << "Multidimensional" << endl;
//cout << m.size() << endl;
//cout << m[2].size() << endl;
//
//cout << "SingleDimensional" << endl;
//
//v[0] = 0.0;
//v[1] = 1.0;
//v[2] = 2.0;
//v[3] = 3.0;
//v[4] = 4.1;
//
//cout << v[4] << endl;
//cout << v.size() << endl;


NeuralNet::NeuralNet(int numInput, int numHidden, int numOutput)
{

	
	srand(time(NULL)); //Seeding Random for first time
	
	this->numInput = numInput;
	this->numHidden = numHidden;
	this->numOutput = numOutput;

	numTestData = 20;//hard coded for now

	ihWeights = MakeMatrix(numInput, numHidden);
	hBiases = new double[numHidden];
	hOutputs = new double[numHidden];

	hoWeights = MakeMatrix(numHidden, numOutput);
	oBiases = new double[numOutput];

	outputs = new double[numOutput];

	hGrads = new double[numHidden];
	oGrads = new double[numOutput];

	ihPrevWeightsDelta = MakeMatrix(numInput, numHidden);
	hPrevBiasesDelta = new double[numHidden];
	hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
	oPrevBiasesDelta = new double[numOutput];

}

NeuralNet::~NeuralNet()
{

	//CLEANING UP POINTERS


	//1 Dimensional Pointers

	if (hBiases != NULL)          delete hBiases;
	if (hOutputs != NULL)         delete hOutputs;
	if (oBiases != NULL)          delete oBiases;
	if (outputs != NULL)          delete outputs;
	if (hGrads != NULL)           delete hGrads;
	if (oGrads != NULL)           delete oGrads;
	if (hPrevBiasesDelta != NULL) delete hPrevBiasesDelta;
	if (oPrevBiasesDelta != NULL) delete oPrevBiasesDelta;

	//2 Dimensional Pointers
	if (ihWeights != NULL)
	{
		for (int i = 0; i < numInput; i++)
			delete[] ihWeights[i];
		delete ihWeights;
	}
	if (hoWeights != NULL)
	{
		for (int i = 0; i < numHidden; i++)
			delete[] hoWeights[i];
		delete hoWeights;
	}
	if (ihPrevWeightsDelta != NULL)
	{
		for (int i = 0; i < numInput; i++)
			delete[] ihPrevWeightsDelta[i];
		delete ihPrevWeightsDelta;
	}
	if (hoPrevWeightsDelta != NULL)
	{
		for (int i = 0; i < numHidden; i++)
			delete[] hoPrevWeightsDelta[i];
		delete hoPrevWeightsDelta;
	}


	system("pause");

}

double** NeuralNet::MakeMatrix(int rows, int cols)
{
	//2 Dimensional Dynamic Array

	double** matrix = new double*[rows];
	for (int i = 0; i < rows; i++)
	{
		matrix[i] = new double[cols];
	}

	return matrix;
}


void NeuralNet::InitializeWeights()
{

	//Initialize Weights and biases to small random values
	int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
	double* initialWeights = new double[numWeights];
	double lo = -0.01;
	double hi = 0.01;

	for (int i = 0; i < numWeights; ++i)
		initialWeights[i] = (hi - lo) * GenerateRand() + lo;
	
	SetWeights(initialWeights);
}

void NeuralNet::SetWeights(double* weights)
{
	//Take the weights and biases from weights array and put them in i-h weights, h-o weights, h-o biases
	int k = 0; //points into weights param

	for (int i = 0; i < numInput; ++i)
		for (int j = 0; j < numHidden; ++j)
			ihWeights[i][j] = weights[k++];
	for (int i = 0; i < numHidden; ++i)
		hBiases[i] = weights[k++];
	for (int i = 0; i < numHidden; ++i)
		for (int j = 0; j < numOutput; ++j)
			hoWeights[i][j] = weights[k++];
	for (int i = 0; i < numOutput; ++i)
		oBiases[i] = weights[k++];
}

double NeuralNet::GenerateRand()
{
	double rnd = ((double)rand() / (RAND_MAX + 1));
	return rnd;
}

void NeuralNet::Train(double** trainData, int maxEpochs, double learnRate, double momentum, double weightDecay)
{
	// train a back-prop style NN classifier using learning rate and momentum
	// weight decay reduces the magnitude of a weight value over time unless that value
	// is constantly increased
	int epoch = 0;
	double* xValues = new double[numInput]; //inputs
	double* tValues = new double[numOutput];//target values

	//Need method of fetching data here - Data then needs to be processed in random order
	int* Sequence = new int[numTestData];
	for (int i = 0; i < numTestData; ++i)
		Sequence[i] = i;

	while (epoch < maxEpochs)
	{
		double mse = MeanSquaredError(trainData);
		if (mse < 0.020) break; // consider passing value in as parameter
		//if (mse < 0.001) break;

		Shuffle(Sequence); // visit each training data in random order

		for (int i = 0; i < numTestData; ++i)
		{
			int idx = Sequence[i];
			CopyArray(trainData[idx], 0, xValues, 0, numInput);
			CopyArray(trainData[idx], numInput, tValues, 0, numOutput);
			ComputeOutputs(xValues); // copy xValues in, compute outputs (store them internally)
			UpdateWeights(tValues, learnRate, momentum, weightDecay); // find better weights
		}// each training tuple
		++epoch;
	}

	
}

void NeuralNet::Shuffle(int* Sequence)
{

	for (int i = 0; i < numTestData; ++i)
	{
		int r = GenerateRand()*numTestData;
		int tmp = Sequence[r];
		Sequence[r] = Sequence[i];
		Sequence[i] = tmp;

	}
}

double NeuralNet::MeanSquaredError(double** TrainData)
{

	//******************Need to finish ComputeOutputs before finishing this function *****************

	//Average squared error per training tuple
	double sumSquaredError = 0.0;
	double* xValues = new double[numInput]; //First numInput values in trainData
	double* tValues = new double[numOutput]; // last numOutput values


	//Walk through each training case, looks like (6.9 3.2 5.7 2.3) (0 0 1)
	//numTestData = size of traindata

	for (int i = 0; i < numTestData; ++i)
	{
		CopyArray(TrainData[i], 0, xValues, 0, numInput);
		CopyArray(TrainData[i], numInput, tValues, 0, numOutput);
		double* yValues = ComputeOutputs(xValues);

		for (int j = 0; j < numOutput; j++)
		{
			double err = tValues[j] - yValues[j];
			sumSquaredError += err*err;
		}

	}
	return sumSquaredError / numTestData;
}

double* NeuralNet::ComputeOutputs(double* xValues)
{
	double* hSums = new double[numHidden]; //hidden nodes sums scratch array
	double* oSums = new double[numOutput]; // output nodes sums

	for (int i = 0; i < numInput; ++i)
		inputs[i] = xValues[i];

	for (int j = 0; j < numHidden; ++j)
		for (int i = 0; i < numInput; ++i)
			hSums[j] += inputs[i] * ihWeights[i][j];

	for (int i = 0; i < numHidden; ++i)
		hSums[i] += hBiases[i];

	for (int i = 0; i < numHidden; ++i) //Apply Activation function
		hOutputs[i] = HyperTanFunction(hSums[i]);

	for (int j = 0; j < numOutput; ++j)
		for (int i = 0; i < numHidden; ++i)
			oSums[j] += hOutputs[i] * hoWeights[i][j];

	for (int i = 0; i < numOutput; ++i)
		oSums[i] += oBiases[i];

	double* softOut = Softmax(oSums); //softmax activation does all outputs at once for efficiency
	CopyArray(softOut, 0, outputs, 0, numOutput);

	double* retResult = new double[numOutput]; // could define a getOutputs method instead
	CopyArray(outputs, 0, retResult, 0, numOutput);

	return retResult;
}

double NeuralNet::HyperTanFunction(double x)
{
	if (x < -20.0) return -1.0; //approximation is correct to 30 decimals
	else if (x > 20.0) return 1.0;
	else return tanh(x);
}

double* NeuralNet::Softmax(double* oSums)
{

	// Determine max output sum
	// does all output nodes at once so scale doesn't have to be re-computed each time

	double max = oSums[0];
	for (int i = 0; i < numOutput; ++i)
		if (oSums[i] > max) max = oSums[i];

	//Determine scaling factor -- sum of exp(each val - max)
	double scale = 0.0;
	for (int i = 0; i < numOutput; ++i)
		scale += exp(oSums[i] - max) / scale;

	double* result = new double[numOutput];
	for (int i = 0; i < numOutput; ++i)
		result[i] = exp(oSums[i] - max) / scale;

	return result; // now scaled so that xi sum to 1.0
}

void NeuralNet::UpdateWeights(double* tValues, double learnRate, double momentum, double weightDecay)
{

	//1. Compute output gradients
	for (int i = 0; i < numOutput; ++i)
	{
		// derivative of softmax = (1 - y) * y (same as log-sigmoid)
		double derivative = (1 - outputs[i]) * outputs[i];
		// mean squared error version include (1-y)(y) derivative
		oGrads[i] = derivative * (tValues[i] - outputs[i]);
	}

	//2. compute hidden gradients
	for (int i = 0; i < numHidden; ++i)
	{
		// derivative of tanh = (1 - y) * (1 + y)
		double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]);
		double sum = 0.0;
		for (int j = 0; j < numOutput; ++j) //each hidden delta is the sum of numOutput terms
		{
			double x = oGrads[j] * hoWeights[i][j];
			sum += x;
		}
		hGrads[i] = derivative * sum;
	}

	//3a. update hidden weights (gradients must be computed right-to-left but weights
	// can be updated in any order)
	for (int i = 0; i < numInput; ++i)
	{

		for (int j = 0; j < numHidden; ++j)
		{

			double delta = learnRate * hGrads[j] * inputs[i]; // compute the new delta
			ihWeights[i][j] += delta; //update. note we use '+' instead of '-' this can be very tricky
									  //now add momentum using previous delta. on first pass old value will be 0.0 but that's ok.
			ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j];
			ihWeights[i][j] -= (weightDecay * ihWeights[i][j]); // weight decay
			ihPrevWeightsDelta[i][j] = delta; // don't forget to save the delta for momentum
		}
	}
		
	//3b. update hidden biases
	for (int i = 0; i < numHidden; ++i)
	{
		double delta = learnRate * hGrads[i] * 1.0; //t1.0 is constant input for bias; could leave out
		hBiases[i] += delta;
		hBiases[i] += momentum * hPrevBiasesDelta[i]; // momentum
		hBiases[i] -= (weightDecay * hBiases[i]); // weight decay
		hPrevBiasesDelta[i] = delta; // don't forget to save the delta for momentum
	}


	//4. update hidden-output weights
	for (int i = 0; i < numHidden; ++i)
	{
		for (int j = 0; j < numOutput; ++j)
		{
			// see above: hOutputs are inputs to the nn outputs
			double delta = learnRate * oGrads[j] * hOutputs[i];
			hoWeights[i][j] += delta;
			hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j]; // momentum
			hoWeights[i][j] -= (weightDecay * hoWeights[i][j]); // weight decay
			hoPrevWeightsDelta[i][j] = delta; // save
		}
	}

	//4b. update output biases
	for (int i = 0; i < numOutput; ++i)
	{
		double delta = learnRate * oGrads[i] * 1.0;
		oBiases[i] += delta;
		oBiases[i] += momentum * oPrevBiasesDelta[i]; // momentum
		oBiases[i] -= (weightDecay * oBiases[i]); // weight decay
		oPrevBiasesDelta[i] = delta; // save
	}
}

void NeuralNet::CopyArray(double* SourceArray, int SourceArrayIdx, double* DestinationArray, int DestinationIndex, int length)
{
	int k = DestinationIndex;
	for (int i = SourceArrayIdx; i < SourceArrayIdx+length; i++)
	{
		SourceArray[i] = DestinationArray[k];
		DestinationIndex++;
	}

}