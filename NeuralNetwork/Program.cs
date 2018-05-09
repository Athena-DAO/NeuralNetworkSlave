using MathNet.Numerics.LinearAlgebra;
using Microsoft.Extensions.Configuration;
using NeuralNetwork.Communication;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;

namespace NeuralNetwork
{
    internal class Program
    {
        public static Matrix<double> ReadCsv(string path)
        {
            StreamReader stream = new StreamReader(path);
            var lines = new List<double[]>();

            while (!stream.EndOfStream)
            {
                string[] line = stream.ReadLine().Split(',');
                var lineValues = new double[line.Length];

                for (int i = 0; i < line.Length; i++)
                {
                    lineValues[i] = double.Parse(line[i]);
                }
                lines.Add(lineValues);
            }

            var data = lines.ToArray();
            stream.Close();

            return Matrix<double>.Build.Dense(data.Length, data[0].Length, (i, j) => data[i][j]);
        }

        public static void WriteCsv(string path, Matrix<double> matrix)
        {
            StreamWriter stream = new StreamWriter(path);

            for (int i = 0; i < matrix.RowCount; i++)
            {
                var result = string.Join(",", matrix.Row(i).ToArray());
                stream.WriteLine(result);
            }
            stream.Close();
        }
        /*
        public static void Main(string[] args)
        {
            //var communicationParameters = JsonConvert.DeserializeObject<CommunicationParameters>(args[1]);
            string pipelineId = args[1];

            CommunicationsLayer communicationLayer = new CommunicationsLayer()
            {
                PipelineId = pipelineId
            };
            communicationLayer.SendCommunicationServerParameters();

            var response = communicationLayer.GetCommunicationResonse();
            bool P2pSuccess = false;
            if (response.P2P)
            {
                IPEndPoint remoteEndPoint = communicationLayer.GetIpEndPoint(response.EndPoint);
                IPEndPoint localEndPoint = communicationLayer.server.client.Client.LocalEndPoint as IPEndPoint;
                communicationLayer.server.Close();
                try
                {
                    TcpHole tcpHole = new TcpHole();
                    TcpClient tcpClient = tcpHole.PunchHole(localEndPoint, remoteEndPoint);

                    if (!tcpHole.Success)
                    {
                        throw new Exception("Hole Punching Failed");
                    }

                    CommunicationModule communicationModule = new CommunicationModule(tcpClient);
                    NeuralNetworkMiddleLayer middleLayer = new NeuralNetworkMiddleLayer() { communicationModule = communicationModule , P2P =true};
                    var neuralNetwork = middleLayer.BuildNeuralNetwork();
                    neuralNetwork.Train();
                    middleLayer.SendTheta(neuralNetwork.Theta);
                    P2pSuccess = true;
                }
                catch (Exception E)
                {
                    if(E.Message!= "Hole Punching Failed")
                    {
                        throw;
                    }
                }
            }


            if (!P2pSuccess)
            {

                CommunicationRabbitMq communicationM2s = new CommunicationRabbitMq() { QueueName = pipelineId + "_" + response.QueueNumber + "m2s" };
                CommunicationRabbitMq communicationS2m = new CommunicationRabbitMq() { QueueName = pipelineId + "_" + response.QueueNumber + "s2m" };
                NeuralNetworkMiddleLayer middleLayer = new NeuralNetworkMiddleLayer() { CommunicationRabbitMqM2s = communicationM2s , CommunicationRabbitMqS2M=communicationS2m , P2P= false };
                var neuralNetwork = middleLayer.BuildNeuralNetwork();
                neuralNetwork.Train();
                middleLayer.SendTheta(neuralNetwork.Theta);


            }
        }
        */
        

        private static void Main(string[] args)
        {
            var Theta1 = ReadCsv("Theta0.csv");
            var Theta2 = ReadCsv("Theta1.csv");
            var X = ReadCsv("X_value.csv");
            var y = ReadCsv("Y_value.csv");
            var x_Test = ReadCsv("X_predict.csv");

            var Theta = new Matrix<double>[2];
            Theta[0] = Theta1;
            Theta[1] = Theta2;
            NeuralNetwork neuralNetwork = new NeuralNetwork()
            {
                X = X,
                y = y,
                Theta = Theta,
                InputLayerSize = 400,
                HiddenLayerSize = 25,
                HiddenLayerLength = 1,
                OutputLayerSize = 10,
                TrainingSize = 5000,
                Lambda = 3,
                Epoch = 50
            };
            var t = neuralNetwork.Cost();
            Console.WriteLine("Cost=", t);

            neuralNetwork.InitializeTheta();
            //neuralNetwork.ReadParams(Theta, X, y);
            neuralNetwork.Train();

            double[] predictions = neuralNetwork.Predict(x_Test);

            WriteCsv("TrainedTheta1.csv", neuralNetwork.Theta[0]);
            WriteCsv("TrainedTheta2.csv", neuralNetwork.Theta[1]);
            Console.ReadLine();
        }

      
    }
}