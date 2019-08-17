#include "NeuralNet.h"

using namespace std;


int main()
{
	ofstream fout;
	fout.open("output.txt");


	vector< vector<double> > allData(150, vector<double>(7));

	fout << "RAW DATA" << endl;

	allData[0] = { 5.1, 3.5, 1.4, 0.2, 0, 0, 1 }; // sepal length, width, petal length, width
	allData[1] = { 4.9, 3.0, 1.4, 0.2, 0, 0, 1 }; // Iris setosa = 0 0 1
	allData[2] = { 4.7, 3.2, 1.3, 0.2, 0, 0, 1 }; // Iris versicolor = 0 1 0
	allData[3] = { 4.6, 3.1, 1.5, 0.2, 0, 0, 1 }; // Iris virginica = 1 0 0
	allData[4] = { 5.0, 3.6, 1.4, 0.2, 0, 0, 1 };
	allData[5] = { 5.4, 3.9, 1.7, 0.4, 0, 0, 1 };
	allData[6] = { 4.6, 3.4, 1.4, 0.3, 0, 0, 1 };
	allData[7] = { 5.0, 3.4, 1.5, 0.2, 0, 0, 1 };
	allData[8] = { 4.4, 2.9, 1.4, 0.2, 0, 0, 1 };
	allData[9] = { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };

	allData[10] = { 5.4, 3.7, 1.5, 0.2, 0, 0, 1 };
	allData[11] = { 4.8, 3.4, 1.6, 0.2, 0, 0, 1 };
	allData[12] = { 4.8, 3.0, 1.4, 0.1, 0, 0, 1 };
	allData[13] = { 4.3, 3.0, 1.1, 0.1, 0, 0, 1 };
	allData[14] = { 5.8, 4.0, 1.2, 0.2, 0, 0, 1 };
	allData[15] = { 5.7, 4.4, 1.5, 0.4, 0, 0, 1 };
	allData[16] = { 5.4, 3.9, 1.3, 0.4, 0, 0, 1 };
	allData[17] = { 5.1, 3.5, 1.4, 0.3, 0, 0, 1 };
	allData[18] = { 5.7, 3.8, 1.7, 0.3, 0, 0, 1 };
	allData[19] = { 5.1, 3.8, 1.5, 0.3, 0, 0, 1 };

	allData[20] = { 5.4, 3.4, 1.7, 0.2, 0, 0, 1 };
	allData[21] = { 5.1, 3.7, 1.5, 0.4, 0, 0, 1 };
	allData[22] = { 4.6, 3.6, 1.0, 0.2, 0, 0, 1 };
	allData[23] = { 5.1, 3.3, 1.7, 0.5, 0, 0, 1 };
	allData[24] = { 4.8, 3.4, 1.9, 0.2, 0, 0, 1 };
	allData[25] = { 5.0, 3.0, 1.6, 0.2, 0, 0, 1 };
	allData[26] = { 5.0, 3.4, 1.6, 0.4, 0, 0, 1 };
	allData[27] = { 5.2, 3.5, 1.5, 0.2, 0, 0, 1 };
	allData[28] = { 5.2, 3.4, 1.4, 0.2, 0, 0, 1 };
	allData[29] = { 4.7, 3.2, 1.6, 0.2, 0, 0, 1 };

	allData[30] = { 4.8, 3.1, 1.6, 0.2, 0, 0, 1 };
	allData[31] = { 5.4, 3.4, 1.5, 0.4, 0, 0, 1 };
	allData[32] = { 5.2, 4.1, 1.5, 0.1, 0, 0, 1 };
	allData[33] = { 5.5, 4.2, 1.4, 0.2, 0, 0, 1 };
	allData[34] = { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
	allData[35] = { 5.0, 3.2, 1.2, 0.2, 0, 0, 1 };
	allData[36] = { 5.5, 3.5, 1.3, 0.2, 0, 0, 1 };
	allData[37] = { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
	allData[38] = { 4.4, 3.0, 1.3, 0.2, 0, 0, 1 };
	allData[39] = { 5.1, 3.4, 1.5, 0.2, 0, 0, 1 };

	allData[40] = { 5.0, 3.5, 1.3, 0.3, 0, 0, 1 };
	allData[41] = { 4.5, 2.3, 1.3, 0.3, 0, 0, 1 };
	allData[42] = { 4.4, 3.2, 1.3, 0.2, 0, 0, 1 };
	allData[43] = { 5.0, 3.5, 1.6, 0.6, 0, 0, 1 };
	allData[44] = { 5.1, 3.8, 1.9, 0.4, 0, 0, 1 };
	allData[45] = { 4.8, 3.0, 1.4, 0.3, 0, 0, 1 };
	allData[46] = { 5.1, 3.8, 1.6, 0.2, 0, 0, 1 };
	allData[47] = { 4.6, 3.2, 1.4, 0.2, 0, 0, 1 };
	allData[48] = { 5.3, 3.7, 1.5, 0.2, 0, 0, 1 };
	allData[49] = { 5.0, 3.3, 1.4, 0.2, 0, 0, 1 };

	allData[50] = { 7.0, 3.2, 4.7, 1.4, 0, 1, 0 };
	allData[51] = { 6.4, 3.2, 4.5, 1.5, 0, 1, 0 };
	allData[52] = { 6.9, 3.1, 4.9, 1.5, 0, 1, 0 };
	allData[53] = { 5.5, 2.3, 4.0, 1.3, 0, 1, 0 };
	allData[54] = { 6.5, 2.8, 4.6, 1.5, 0, 1, 0 };
	allData[55] = { 5.7, 2.8, 4.5, 1.3, 0, 1, 0 };
	allData[56] = { 6.3, 3.3, 4.7, 1.6, 0, 1, 0 };
	allData[57] = { 4.9, 2.4, 3.3, 1.0, 0, 1, 0 };
	allData[58] = { 6.6, 2.9, 4.6, 1.3, 0, 1, 0 };
	allData[59] = { 5.2, 2.7, 3.9, 1.4, 0, 1, 0 };

	allData[60] = { 5.0, 2.0, 3.5, 1.0, 0, 1, 0 };
	allData[61] = { 5.9, 3.0, 4.2, 1.5, 0, 1, 0 };
	allData[62] = { 6.0, 2.2, 4.0, 1.0, 0, 1, 0 };
	allData[63] = { 6.1, 2.9, 4.7, 1.4, 0, 1, 0 };
	allData[64] = { 5.6, 2.9, 3.6, 1.3, 0, 1, 0 };
	allData[65] = { 6.7, 3.1, 4.4, 1.4, 0, 1, 0 };
	allData[66] = { 5.6, 3.0, 4.5, 1.5, 0, 1, 0 };
	allData[67] = { 5.8, 2.7, 4.1, 1.0, 0, 1, 0 };
	allData[68] = { 6.2, 2.2, 4.5, 1.5, 0, 1, 0 };
	allData[69] = { 5.6, 2.5, 3.9, 1.1, 0, 1, 0 };

	allData[70] = { 5.9, 3.2, 4.8, 1.8, 0, 1, 0 };
	allData[71] = { 6.1, 2.8, 4.0, 1.3, 0, 1, 0 };
	allData[72] = { 6.3, 2.5, 4.9, 1.5, 0, 1, 0 };
	allData[73] = { 6.1, 2.8, 4.7, 1.2, 0, 1, 0 };
	allData[74] = { 6.4, 2.9, 4.3, 1.3, 0, 1, 0 };
	allData[75] = { 6.6, 3.0, 4.4, 1.4, 0, 1, 0 };
	allData[76] = { 6.8, 2.8, 4.8, 1.4, 0, 1, 0 };
	allData[77] = { 6.7, 3.0, 5.0, 1.7, 0, 1, 0 };
	allData[78] = { 6.0, 2.9, 4.5, 1.5, 0, 1, 0 };
	allData[79] = { 5.7, 2.6, 3.5, 1.0, 0, 1, 0 };

	allData[80] = { 5.5, 2.4, 3.8, 1.1, 0, 1, 0 };
	allData[81] = { 5.5, 2.4, 3.7, 1.0, 0, 1, 0 };
	allData[82] = { 5.8, 2.7, 3.9, 1.2, 0, 1, 0 };
	allData[83] = { 6.0, 2.7, 5.1, 1.6, 0, 1, 0 };
	allData[84] = { 5.4, 3.0, 4.5, 1.5, 0, 1, 0 };
	allData[85] = { 6.0, 3.4, 4.5, 1.6, 0, 1, 0 };
	allData[86] = { 6.7, 3.1, 4.7, 1.5, 0, 1, 0 };
	allData[87] = { 6.3, 2.3, 4.4, 1.3, 0, 1, 0 };
	allData[88] = { 5.6, 3.0, 4.1, 1.3, 0, 1, 0 };
	allData[89] = { 5.5, 2.5, 4.0, 1.3, 0, 1, 0 };

	allData[90] = { 5.5, 2.6, 4.4, 1.2, 0, 1, 0 };
	allData[91] = { 6.1, 3.0, 4.6, 1.4, 0, 1, 0 };
	allData[92] = { 5.8, 2.6, 4.0, 1.2, 0, 1, 0 };
	allData[93] = { 5.0, 2.3, 3.3, 1.0, 0, 1, 0 };
	allData[94] = { 5.6, 2.7, 4.2, 1.3, 0, 1, 0 };
	allData[95] = { 5.7, 3.0, 4.2, 1.2, 0, 1, 0 };
	allData[96] = { 5.7, 2.9, 4.2, 1.3, 0, 1, 0 };
	allData[97] = { 6.2, 2.9, 4.3, 1.3, 0, 1, 0 };
	allData[98] = { 5.1, 2.5, 3.0, 1.1, 0, 1, 0 };
	allData[99] = { 5.7, 2.8, 4.1, 1.3, 0, 1, 0 };

	allData[100] = { 6.3, 3.3, 6.0, 2.5, 1, 0, 0 };
	allData[101] = { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
	allData[102] = { 7.1, 3.0, 5.9, 2.1, 1, 0, 0 };
	allData[103] = { 6.3, 2.9, 5.6, 1.8, 1, 0, 0 };
	allData[104] = { 6.5, 3.0, 5.8, 2.2, 1, 0, 0 };
	allData[105] = { 7.6, 3.0, 6.6, 2.1, 1, 0, 0 };
	allData[106] = { 4.9, 2.5, 4.5, 1.7, 1, 0, 0 };
	allData[107] = { 7.3, 2.9, 6.3, 1.8, 1, 0, 0 };
	allData[108] = { 6.7, 2.5, 5.8, 1.8, 1, 0, 0 };
	allData[109] = { 7.2, 3.6, 6.1, 2.5, 1, 0, 0 };

	allData[110] = { 6.5, 3.2, 5.1, 2.0, 1, 0, 0 };
	allData[111] = { 6.4, 2.7, 5.3, 1.9, 1, 0, 0 };
	allData[112] = { 6.8, 3.0, 5.5, 2.1, 1, 0, 0 };
	allData[113] = { 5.7, 2.5, 5.0, 2.0, 1, 0, 0 };
	allData[114] = { 5.8, 2.8, 5.1, 2.4, 1, 0, 0 };
	allData[115] = { 6.4, 3.2, 5.3, 2.3, 1, 0, 0 };
	allData[116] = { 6.5, 3.0, 5.5, 1.8, 1, 0, 0 };
	allData[117] = { 7.7, 3.8, 6.7, 2.2, 1, 0, 0 };
	allData[118] = { 7.7, 2.6, 6.9, 2.3, 1, 0, 0 };
	allData[119] = { 6.0, 2.2, 5.0, 1.5, 1, 0, 0 };

	allData[120] = { 6.9, 3.2, 5.7, 2.3, 1, 0, 0 };
	allData[121] = { 5.6, 2.8, 4.9, 2.0, 1, 0, 0 };
	allData[122] = { 7.7, 2.8, 6.7, 2.0, 1, 0, 0 };
	allData[123] = { 6.3, 2.7, 4.9, 1.8, 1, 0, 0 };
	allData[124] = { 6.7, 3.3, 5.7, 2.1, 1, 0, 0 };
	allData[125] = { 7.2, 3.2, 6.0, 1.8, 1, 0, 0 };
	allData[126] = { 6.2, 2.8, 4.8, 1.8, 1, 0, 0 };
	allData[127] = { 6.1, 3.0, 4.9, 1.8, 1, 0, 0 };
	allData[128] = { 6.4, 2.8, 5.6, 2.1, 1, 0, 0 };
	allData[129] = { 7.2, 3.0, 5.8, 1.6, 1, 0, 0 };

	allData[130] = { 7.4, 2.8, 6.1, 1.9, 1, 0, 0 };
	allData[131] = { 7.9, 3.8, 6.4, 2.0, 1, 0, 0 };
	allData[132] = { 6.4, 2.8, 5.6, 2.2, 1, 0, 0 };
	allData[133] = { 6.3, 2.8, 5.1, 1.5, 1, 0, 0 };
	allData[134] = { 6.1, 2.6, 5.6, 1.4, 1, 0, 0 };
	allData[135] = { 7.7, 3.0, 6.1, 2.3, 1, 0, 0 };
	allData[136] = { 6.3, 3.4, 5.6, 2.4, 1, 0, 0 };
	allData[137] = { 6.4, 3.1, 5.5, 1.8, 1, 0, 0 };
	allData[138] = { 6.0, 3.0, 4.8, 1.8, 1, 0, 0 };
	allData[139] = { 6.9, 3.1, 5.4, 2.1, 1, 0, 0 };

	allData[140] = { 6.7, 3.1, 5.6, 2.4, 1, 0, 0 };
	allData[141] = { 6.9, 3.1, 5.1, 2.3, 1, 0, 0 };
	allData[142] = { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
	allData[143] = { 6.8, 3.2, 5.9, 2.3, 1, 0, 0 };
	allData[144] = { 6.7, 3.3, 5.7, 2.5, 1, 0, 0 };
	allData[145] = { 6.7, 3.0, 5.2, 2.3, 1, 0, 0 };
	allData[146] = { 6.3, 2.5, 5.0, 1.9, 1, 0, 0 };
	allData[147] = { 6.5, 3.0, 5.2, 2.0, 1, 0, 0 };
	allData[148] = { 6.2, 3.4, 5.4, 2.3, 1, 0, 0 };
	allData[149] = { 5.9, 3.0, 5.1, 1.8, 1, 0, 0 };

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