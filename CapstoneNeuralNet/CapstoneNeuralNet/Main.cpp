#include "NeuralNet.h"

using namespace std;


int main()
{
	ofstream fout;
	fout.open("output.txt");
	int DEBUG = 1;

	vector< vector<double> > allData(150, vector<double>(7));

	fout << "RAW DATA" << endl;

	

	TickerObject BMO("C:\\Users\\Massimo\\Documents\\Programming_Projects\\NeuralNetwork\\CapstoneNeuralNet\\Data\\BCE.TO.csv");

	cout << "high on jan 1st 2015: " << BMO.Get_High(2019, 6, 12) << endl;

	//long lastDate;
	//lastDate = epoch_time(1996, 5, 6, 1);
	//cout << lastDate << endl;

	system("pause");
	
	if (DEBUG == 1) return 0;

	//Displaying Data
	for (int i = 0; i < 150; i++)
	{
		fout << "Data[" << i << "],";
		for (int j = 0; j < 7; j++)
			fout << allData[i][j] << ",";

		fout << endl;
	}

	//Need a parse data function here to read inputs from text file and to delimit the data

	vector< vector<double> > TrainData(1, vector<double>(1));	//Simple initialization of Training Data array and [1,1]
	vector< vector<double> > TestData(1, vector<double>(1));	//Simple initialization of Testing Data array [1,1] 

	NeuralNet nn(4, 7, 3);

	nn.MakeTrainTest(allData, TrainData, TestData);				//Split the total data and fill into Training Data and Testing Data - standard is 80-20 split


	vector<int> Cols = { 0, 1, 2, 3 };							//Specifying which columns in the Data needs to be normalized

	nn.Normalize(TrainData, Cols);								//Normalize the Training Data
	nn.Normalize(TestData, Cols);								//Normalize the Testing Data


	nn.InitializeWeights();

	int maxEpochs = 2000;//2000
	double learnRate = 0.05;//0.05
	double momentum = 0.01;//0.01
	double weightDecay = 0.0001;//0.0001

	nn.Train(TrainData, maxEpochs, learnRate, momentum, weightDecay);

	double trainAcc = nn.Accuracy(TrainData);

	double testAcc = nn.Accuracy(TestData);

	cout << "Train Accuracy: " << trainAcc << endl;
	cout << "Test Accuracy = " << testAcc << endl;



	vector<double> weights;

	nn.GetWeights(weights);

	for (size_t i = 0; i < weights.size(); i++)
		cout << weights[i] << "      ";


	fout.close();
	vector<double> input(4);
	input = { 5.8, 3.1, 5.0, 1.8 };
	vector<double> output(3);

	output = nn.ComputeOutputs(input);

	cout << endl;

	for (int i = 0; i < 3; i++)//should equal to 1 0 0
		cout << output[i] << "  " << endl;

	system("pause");
	return 0;
}


//1-Data acquisition — this provides us the features
//2-Data preprocessing — an often dreaded but necessary step to make the data usable
//3-Develop and implement model — where we choose the type of neural network and parameters
//4-Backtest model — a very crucial step in any trading strategy
//5-Optimization — finding suitable parameters