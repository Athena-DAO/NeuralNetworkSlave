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

            return Matrix<double>.Build.Dense(data.Length, data[0].Length, (i, j) => data[i][j]);

        }





        static void Main(string[] args)
        {
            var Theta1 = ReadCsv("Theta1_value.csv");
            var Theta2 = ReadCsv("Theta2_value.csv");
            var X = ReadCsv("X_value.csv");
            var y = ReadCsv("Y_value.csv");
            Matrix<double>[] Theta = new Matrix<double>[2];
            Theta[0] = Theta1;
            Theta[1] = Theta2;
            NeuralNetwork neuralNetwork = new NeuralNetwork()
            {
                InputLayerSize = 400,
                HiddenLayerSize = 25,
                HiddenLayerLength = 1,
                OutputLayerSize = 10,
                TrainingSize = 5000,
                Lambda = 1
            };

            neuralNetwork.ReadParams(Theta, X, y);

            double ct = neuralNetwork.Cost();

            Console.ReadLine();
        }
    }
}
