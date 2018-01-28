using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
namespace NeuralNetwork
{
    class Program
    {

        public static Matrix<double> ReadCsv(string path)
        {
            StreamReader stream = new StreamReader(path);
            var lines = new List<double[]>();

            while (!stream.EndOfStream)
            {
                string[] line = stream.ReadLine().Split(',');
                var lineValues = new double[line.Length];

                for (int i=0;i<line.Length;i++)
                {
                    lineValues[i] = double.Parse(line[i]);
                }
                lines.Add(lineValues);
            }

            var data = lines.ToArray();
            stream.Close();

            return Matrix<double>.Build.Dense(data.Length, data[0].Length, (i, j) => data[i][j]);

            
        }



        public static void WriteCsv(string path , Matrix<double> matrix)
        {
            StreamWriter stream = new StreamWriter(path);


          for(int i=0;i<matrix.RowCount;i++)
            {
                var result = string.Join(",", matrix.Row(i).ToArray());
                stream.WriteLine(result);
            }
            stream.Close();
        }




        static void Main(string[] args)
        {
            var Theta1 = ReadCsv("Theta1_value.csv");
            var Theta2 = ReadCsv("Theta2_value.csv");
            var X = ReadCsv("X_value.csv");
            var y = ReadCsv("Y_value.csv");
            var x_Test = ReadCsv("X_predict.csv");

            WriteCsv("Hello.csv",Theta1);

            Matrix<double>[] Theta = new Matrix<double>[2];
            Theta[0] = Matrix<double>.Build.Random(25, 401);
            Theta[1] = Matrix<double>.Build.Random(10, 26);
            NeuralNetwork neuralNetwork = new NeuralNetwork()
            {
                InputLayerSize = 400,
                HiddenLayerSize = 25,
                HiddenLayerLength = 1,
                OutputLayerSize = 10,
                TrainingSize = 5000,
                Lambda = 3
            };

            neuralNetwork.ReadParams(Theta, X, y);

            neuralNetwork.Train(200);



            double[] predictions = neuralNetwork.predict(x_Test);

            WriteCsv("TrainedTheta1.csv", neuralNetwork.Theta[0]);
            WriteCsv("TrainedTheta2.csv", neuralNetwork.Theta[1]);
            Console.ReadLine();
        }
    }
}
