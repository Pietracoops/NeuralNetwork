#include "NeuralNet.h"

using namespace std;


int main()
{


//========================================================== DEBUGGING SECTION ============================================================
	ofstream fout;
	fout.open("output.txt");
	int DEBUG = 0;

	vector< vector<double> > allData(150, vector<double>(7));

	fout << "RAW DATA" << endl;

	string filename = "C:\\Users\\Massimo\\Documents\\Programming_Projects\\NeuralNetwork\\CapstoneNeuralNet\\Data\\BCE.TO.csv";

	//IF
	TickerObject BMO(filename, true);
	int datasize = BMO.Get_Data_Size();
	int dataset = 7;

	vector< vector<double> > stockData(datasize, vector<double>(dataset));
	//stockData = BMO.Make_Training_Data();

	if (DEBUG == 1) system("pause");
	if (DEBUG == 1) return 0;


//=========================================================================================================================================
	vector<double> dubarray;
	double diviser = 100;
	dubarray = { 1, 1 / diviser };
	stockData.push_back(dubarray);
	dubarray = { 2, 2 / diviser };
	stockData.push_back(dubarray);
	dubarray = { 3, 3 / diviser };
	stockData.push_back(dubarray);
	dubarray = { 4, 4 / diviser };
	stockData.push_back(dubarray);
	dubarray = { 5, 5 / diviser };
	stockData.push_back(dubarray);
	dubarray = { 6, 6 / diviser };
	stockData.push_back(dubarray);
	dubarray = { 7, 7 / diviser };
	stockData.push_back(dubarray);
	dubarray = { 8, 8 / diviser };
	stockData.push_back(dubarray);
	dubarray = { 9, 9 / diviser };
	stockData.push_back(dubarray);
	dubarray = { 10, 10 / diviser };
	stockData.push_back(dubarray);


	//Need a parse data function here to read inputs from text file and to delimit the data

	vector< vector<double> > TrainData(1, vector<double>(1));	//Simple initialization of Training Data array and [1,1]
	vector< vector<double> > TestData(1, vector<double>(1));	//Simple initialization of Testing Data array [1,1] 

	NeuralNet nn(1, 2, 1);

	nn.MakeTrainTest(stockData, TrainData, TestData);			//Split the total data and fill into Training Data and Testing Data - standard is 80-20 split


	vector<int> Cols = { 0 };							//Specifying which columns in the Data needs to be normalized

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
	vector<double> input;
	input.push_back(11);
	vector<double> output(1);

	output = nn.ComputeOutputs(input);

	cout << endl;

	double answer = output[0] * diviser;
	cout << answer << endl;

	system("pause");
	return 0;
}


//1-Data acquisition — this provides us the features
//2-Data preprocessing — an often dreaded but necessary step to make the data usable
//3-Develop and implement model — where we choose the type of neural network and parameters
//4-Backtest model — a very crucial step in any trading strategy
//5-Optimization — finding suitable parameters