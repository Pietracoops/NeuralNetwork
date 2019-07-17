#include "NeuralNet.h"
using namespace std;

ofstream fout;

NeuralNet::NeuralNet(int numInput, int numHidden, int numOutput)
{
	//Constructor

	numTestData = 20;							//hard coded for now
	srand((unsigned int)time(NULL));							//Seeding Random for first time

	this->numInput = numInput;					//Initializing # of Input Nodes
	this->numHidden = numHidden;				//Initializing # of Hidden Nodes
	this->numOutput = numOutput;				//Initializing # of Output Nodes


	inputs.resize(numInput);					//Creating Inputs Array
	ihWeights.resize(numInput);					//Creating 1st layer of multidimensional array (number of inputs)
	hBiases.resize(numHidden);					//Initializing Biases array (1 for each hidden node)
	for (int i = 0; i < numInput; ++i)			//Creating Multidimensional Weights Array where first array index is # of Input node and second is # of Weight
		ihWeights[i].resize(numHidden);			

	hOutputs.resize(numHidden);					//Initializing Outputs Array
	hoWeights.resize(numHidden);				//Initializing first layer of multidimensional array (number of hidden nodes)
	for (int i = 0; i < numHidden; ++i)			//Creating Multidimensional Weights Array where first array index is # of Input node and second is # of Weight
		hoWeights[i].resize(numOutput);			
	

	outputs.resize(numOutput);					//Creating Outputs Array
	oBiases.resize(numOutput);					//Creating Output Biases Array

	hGrads.resize(numHidden);					//Resizing Gradients 
	oGrads.resize(numOutput);					//Resizing Gradients


	ihPrevWeightsDelta.resize(numInput);
	for (int i = 0; i < numInput; ++i)
		ihPrevWeightsDelta[i].resize(numHidden);
	hPrevBiasesDelta.resize(numHidden);
	hoPrevWeightsDelta.resize(numHidden);
	for (int i = 0; i < numHidden; ++i)
		hoPrevWeightsDelta[i].resize(numOutput);
	oPrevBiasesDelta.resize(numOutput);


	fout.open("Data.txt");
}

NeuralNet::~NeuralNet()
{
	//Destructor
	fout.close();

	system("pause");

}

double** NeuralNet::MakeMatrix(int rows, int cols)// Retired - this was used for dynamic arrays using pointers
{
	//2 Dimensional Dynamic Array

	double** matrix = new double*[rows];
	for (int i = 0; i < rows; i++)
	{
		matrix[i] = new double[cols];
	}

	return matrix;
}


void NeuralNet::MakeTrainTest(std::vector< std::vector<double> >& AllData, std::vector< std::vector<double> >& TrainData, std::vector< std::vector<double> >& TestData)
{
	//Split AllData into 80% trainData and 20% testData
	int totRows = AllData.size();
	int numCols = AllData[0].size();

	int trainRows = (int)(totRows*0.80);//hard coded 80-20 split
	int testRows = totRows - trainRows;

	vector<double> trainData;
	vector<double> testData;

	TrainData.resize(trainRows);
	for (int i = 0; i < trainRows; i++)
	{
		TrainData[i].resize(numCols);
	}
	TestData.resize(testRows);
	for (int i = 0; i < testRows; i++)
	{
		TestData[i].resize(numCols);
	}


	vector<int> sequence(totRows);

	for (size_t i = 0; i < sequence.size(); ++i)
		sequence[i] = i;

	for (size_t i = 0; i < sequence.size(); ++i)
	{
		int r = (int)(GenerateRand() * sequence.size());
		int tmp = sequence[r];
		sequence[r] = sequence[i];
		sequence[i] = tmp;
	}

	int si = 0; // index into sequence[]
	int j = 0; // index into trainData or testData
	int idx = 0;
	for (; si < trainRows; ++si)
	{
		idx = sequence[si];
		CopyArray(AllData[idx], 0, TrainData[j], 0, numCols);
		++j;
	}

	j = 0;

	for (; si < totRows; ++si)
	{
		idx = sequence[si];
		CopyArray(AllData[idx], 0, TestData[j], 0, numCols);
		++j;
	}
}

void NeuralNet::Normalize(std::vector< std::vector<double> >& DataMatrix, std::vector<int> cols)
{
	//normalize specified cols by computing (x - mean) / sd for each value

	for each (int col in cols)
	{
		double sum = 0.0;

		for (size_t i = 0; i < DataMatrix.size(); ++i)
			sum += DataMatrix[i][col];
		double mean = sum / DataMatrix.size();
		sum = 0.0;
		for (size_t i = 0; i < DataMatrix.size(); ++i)
			sum += (DataMatrix[i][col] - mean) * (DataMatrix[i][col] - mean);
		double sd = sqrt(sum / (DataMatrix.size() - 1));
		for (size_t i = 0; i < DataMatrix.size(); ++i)
			DataMatrix[i][col] = (DataMatrix[i][col] - mean) / sd;
	}
}


void NeuralNet::InitializeWeights()
{

	//Initialize Weights and biases to small random values
	int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
	vector<double> initialWeights(numWeights);

	double lo = -0.01;			//------(-0.01)---|---(0.01)------
	double hi = 0.01;

	for (size_t i = 0; i < initialWeights.size(); ++i)
		initialWeights[i] = (hi - lo) * GenerateRand() + lo;

	SetWeights(initialWeights);
}

void NeuralNet::ExportWeights(string filename)
{
	//ofstream fout(filename);
	//
	//
	//
	//
	//fout.close();
	//
	//
	//// returns the current set of weights, presumably after training
	//int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
	//result.resize(numWeights);
	//int k = 0;
	//
	//for (int i = 0; i < ihWeights.size(); ++i)
	//	for (int j = 0; j < ihWeights[0].size(); ++j)
	//		result[k++] = ihWeights[i][j];
	//for (int i = 0; i < hBiases.size(); ++i)
	//	result[k++] = hBiases[i];
	//for (int i = 0; i < hoWeights.size(); ++i)
	//	for (int j = 0; j < hoWeights[0].size(); ++j)
	//		result[k++] = hoWeights[i][j];
	//for (int i = 0; i < oBiases.size(); ++i)
	//	result[k++] = oBiases[i];
}


void NeuralNet::GetWeights(vector<double>& result)
{
	// returns the current set of weights, presumably after training
	int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
	result.resize(numWeights);
	int k = 0;

	for (size_t i = 0; i < ihWeights.size(); ++i)
		for (size_t j = 0; j < ihWeights[0].size(); ++j)
			result[k++] = ihWeights[i][j];
	for (size_t i = 0; i < hBiases.size(); ++i)
		result[k++] = hBiases[i];
	for (size_t i = 0; i < hoWeights.size(); ++i)
		for (size_t j = 0; j < hoWeights[0].size(); ++j)
			result[k++] = hoWeights[i][j];
	for (size_t i = 0; i < oBiases.size(); ++i)
		result[k++] = oBiases[i];
}


void NeuralNet::SetWeights(vector<double> weights)
{
	numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
	if (weights.size() != numWeights)
	{
		cout << "Bad weights array length" << endl;
		exit(0);
	}

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


void NeuralNet::Train(vector< vector<double> >& trainData, int maxEpochs, double learnRate, double momentum, double weightDecay)
{
	// train a back-prop style NN classifier using learning rate and momentum
	// weight decay reduces the magnitude of a weight value over time unless that value
	// is constantly increased
	int epoch = 0;
	vector<double> xValues(numInput);
	vector<double> tValues(numOutput);

	float PercentageComp = 0.0;
	//Need method of fetching data here - Data then needs to be processed in random order
	vector<int> Sequence(trainData.size());
	for (size_t i = 0; i < trainData.size(); ++i)
		Sequence[i] = i;

	while (epoch < maxEpochs)
	{
		double mse = MeanSquaredError(trainData);
		if (mse < 0.020) break; // consider passing value in as parameter
								//if (mse < 0.001) break;

		Shuffle(Sequence); // visit each training data in random order

		for (size_t i = 0; i < trainData.size(); ++i)
		{
			int idx = Sequence[i];
			CopyArray(trainData[idx], 0, xValues, 0, numInput);
			CopyArray(trainData[idx], numInput, tValues, 0, numOutput);
			ComputeOutputs(xValues); // copy xValues in, compute outputs (store them internally)
			UpdateWeights(tValues, learnRate, momentum, weightDecay); // find better weights
			StoreDatabase(0);
		}// each training tuple
		++epoch;

		PercentageComp = ((float)epoch / (float)maxEpochs) * 100;
		cout << "Percentage Complete: " << PercentageComp << endl;
	}

	StoreDatabase(1);
}

void NeuralNet::Shuffle(vector<int>& Sequence)
{

	for (int i = 0; i < numTestData; ++i)
	{
		int r = (int)GenerateRand()*numTestData;
		int tmp = Sequence[r];
		Sequence[r] = Sequence[i];
		Sequence[i] = tmp;

	}
}

void NeuralNet::StoreDatabase(int i)
{

	if (i == 0)
	{
		fout << "Input:,";

		for (size_t i = 0; i < inputs.size(); i++)
			fout << inputs[i] << ",";


		fout << ",Outputs:,";
		for (size_t i = 0; i < outputs.size(); i++)
			fout << outputs[i] << ",";

		fout << endl;
	}
	else if (i == 1)
	{
		fout << endl;


		int k = 0;

		for (size_t i = 0; i < ihWeights.size(); ++i)
			for (size_t j = 0; j < ihWeights[0].size(); ++j)
				fout << ihWeights[i][j] << endl;
		for (size_t i = 0; i < hBiases.size(); ++i)
			fout << hBiases[i] << endl;
		for (size_t i = 0; i < hoWeights.size(); ++i)
			for (size_t j = 0; j < hoWeights[0].size(); ++j)
				fout << hoWeights[i][j] << endl;
		for (size_t i = 0; i < oBiases.size(); ++i)
			fout << oBiases[i] << endl;

	}




}

double NeuralNet::MeanSquaredError(vector< vector<double> >& TrainData)
{


	//Average squared error per training tuple
	double sumSquaredError = 0.0;

	vector<double> xValues(numInput); //First numInput values in trainData
	vector<double> tValues(numOutput);// last numOutput values


									  //Walk through each training case, looks like (6.9 3.2 5.7 2.3) (0 0 1)
									  //numTestData = size of traindata

	for (size_t i = 0; i < TrainData.size(); ++i)
	{
		CopyArray(TrainData[i], 0, xValues, 0, numInput);
		CopyArray(TrainData[i], numInput, tValues, 0, numOutput);
		vector<double> yValues = ComputeOutputs(xValues);

		for (int j = 0; j < numOutput; j++)
		{
			double err = tValues[j] - yValues[j];
			sumSquaredError += err*err;
		}

	}
	return sumSquaredError / numTestData;
}

vector<double> NeuralNet::ComputeOutputs(vector<double> xValues)
{
	vector<double> hSums(numHidden); //hidden nodes sums scratch array
	vector<double> oSums(numOutput); // output nodes sums

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

	vector<double> softOut = Softmax(oSums); //softmax activation does all outputs at once for efficiency
	CopyArray(softOut, 0, outputs, 0, numOutput);

	vector<double> retResult(numOutput); // could define a getOutputs method instead
	CopyArray(outputs, 0, retResult, 0, numOutput);

	return retResult;
}

double NeuralNet::HyperTanFunction(double x)
{
	if (x < -20.0) return -1.0; //approximation is correct to 30 decimals
	else if (x > 20.0) return 1.0;
	else return tanh(x);
}

vector<double> NeuralNet::Softmax(vector<double> oSums)
{

	// Determine max output sum
	// does all output nodes at once so scale doesn't have to be re-computed each time

	double max = oSums[0];
	for (size_t i = 0; i < oSums.size(); ++i)
	{
		if (oSums[i] > max) max = oSums[i];
	}


	//Determine scaling factor -- sum of exp(each val - max)
	double scale = 0.0;
	for (size_t i = 0; i < oSums.size(); ++i)
		scale += exp(oSums[i] - max);

	vector<double> result(oSums.size());
	for (size_t i = 0; i < oSums.size(); ++i)
		result[i] = exp(oSums[i] - max) / scale;

	return result; // now scaled so that xi sum to 1.0
}

void NeuralNet::UpdateWeights(vector<double> tValues, double learnRate, double momentum, double weightDecay)
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

double NeuralNet::Accuracy(std::vector< std::vector<double> >& testData)
{

	// percentage correct using winner-takes all
	int numCorrect = 0;
	int numWrong = 0;
	double result;
	vector<double> xValues(numInput); //inputs
	vector<double> tValues(numOutput); //targets
	vector<double> yValues; //computed Y



	for (size_t i = 0; i < testData.size(); ++i)
	{
		CopyArray(testData[i], 0, xValues, 0, numInput);
		CopyArray(testData[i], numInput, tValues, 0, numOutput);
		yValues = ComputeOutputs(xValues);
		int maxIndex = MaxIndex(yValues); //which cell in yValue has largest value?

		if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y)
			++numCorrect;
		else
			++numWrong;
	}

	if (numCorrect + numWrong == 0.0)
	{
		cout << "[~] Accuracy function: Division by 0 not allowed." << endl;
		exit(0);
	}

	result = (numCorrect *1.0) / (numCorrect + numWrong);

	return result;
}


int NeuralNet::MaxIndex(std::vector<double> vector)
{

	// index of largest value
	int bigIndex = 0;
	double biggestVal = vector[0];

	for (size_t i = 0; i < vector.size(); ++i)
	{
		if (vector[i] > biggestVal)
		{
			biggestVal = vector[i];
			bigIndex = i;
		}
	}
	return bigIndex;
}



void NeuralNet::CopyArray(vector<double>& SourceArray, int SourceArrayIdx, vector<double>& DestinationArray, int DestinationIndex, int length)
{
	int k = DestinationIndex;
	for (int i = SourceArrayIdx; i < SourceArrayIdx + length; i++)
	{
		DestinationArray[k] = SourceArray[i];
		k++;
	}

}