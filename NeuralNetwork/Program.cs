using MathNet.Numerics.LinearAlgebra;
using Microsoft.Extensions.Configuration;
using NeuralNetwork.Communication;
using NeuralNetwork.Logging;
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
        public static IConfiguration BuildConfiguration()
        {
            var builder = new ConfigurationBuilder().SetBasePath(Directory.GetCurrentDirectory()).AddJsonFile("appsettings.json");
            return builder.Build();
        }
        public static void Main(string[] args)
        {
            string pipelineId = args[1];

            IConfiguration Configuration = BuildConfiguration();

            CommunicationsServer communicationServer = new CommunicationsServer(Configuration)
            {
                PipelineId = pipelineId
            };

            communicationServer.SendCommunicationServerParameters();

            var response = communicationServer.GetCommunicationResonse();
            bool P2pSuccess = false;
            NeuralNetworkMiddleLayer middleLayer=null;

            if (response.P2P)
            {
                IPEndPoint remoteEndPoint = communicationServer.GetIpEndPoint(response.EndPoint);
                IPEndPoint localEndPoint = communicationServer.server.client.Client.LocalEndPoint as IPEndPoint;
                communicationServer.server.Close();
                try
                {
                    TcpHole tcpHole = new TcpHole();
                    TcpClient tcpClient = tcpHole.PunchHole(localEndPoint, remoteEndPoint);

                    if (!tcpHole.Success)
                    {
                        throw new Exception("Hole Punching Failed");
                    }

                    CommunicationTcp communicationTcp = new CommunicationTcp(tcpClient);
                    middleLayer = new NeuralNetworkMiddleLayer()
                    {
                        CommunicationModule = new CommunicationModule()
                        {
                            CommunicationTcp = communicationTcp,
                            P2P = true
                        }
                    };
                    P2pSuccess = true;
                }
                catch (Exception E)
                {
                    if (E.Message != "Hole Punching Failed")
                    {
                        throw;
                    }
                }
            }

            if (!P2pSuccess)
            {
                CommunicationRabbitMq communicationM2s = new CommunicationRabbitMq (pipelineId + "_" + response.QueueNumber + "m2s" ,Configuration);
                communicationM2s.StartConsumer();
                CommunicationRabbitMq communicationS2m = new CommunicationRabbitMq(pipelineId + "_" + response.QueueNumber + "s2m" ,Configuration);
                middleLayer = new NeuralNetworkMiddleLayer()
                {
                    CommunicationModule = new CommunicationModule()
                    {
                        CommunicationRabbitMqM2S = communicationM2s,
                        CommunicationRabbitMqS2M = communicationS2m,
                        P2P = false
                    }
                };
            }
            var neuralNetwork = middleLayer.BuildNeuralNetwork();
            var logService = new LogService() { communicationModule = middleLayer.CommunicationModule ,Configuration=Configuration};
            neuralNetwork.LogService = logService;
            neuralNetwork.Configuration = Configuration;
            try
            {
                logService.StartLogService();
                neuralNetwork.Train();
                logService.StopLogService();
                middleLayer.SendTheta(neuralNetwork.Theta);
            }catch(Exception e)
            {
                logService.AddLog("error", "Training Failed");
            }
            middleLayer.CommunicationModule.Close();
        }
        
        
        /*
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
            //var t = neuralNetwork.Cost();
            //Console.WriteLine("Cost=", t);

            neuralNetwork.InitializeTheta();
            //neuralNetwork.ReadParams(Theta, X, y);
            neuralNetwork.Train();

            double[] predictions = neuralNetwork.Predict(x_Test);

            WriteCsv("TrainedTheta1.csv", neuralNetwork.Theta[0]);
            WriteCsv("TrainedTheta2.csv", neuralNetwork.Theta[1]);
            Console.ReadLine();
        }
        */
    }
}