using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Communication;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace NeuralNetwork
{
    internal class NeuralNetworkMiddleLayer
    {

        public CommunicationModule CommunicationModule { get; set; }

        public NeuralNetwork BuildNeuralNetwork()
        {
            string json = CommunicationModule.ReceiveData();
            
            /*
            if (P2P)
            {
               json = communicationTcp.ReceiveData();
            }else
            {
                json = CommunicationRabbitMqM2s.Consume();
            }
            */
            NeuralNetworkParameters neuralNetworkParameters = JsonConvert.DeserializeObject<NeuralNetworkParameters>(json);
            var X = BuildMatrix(neuralNetworkParameters.XDataSize);
            var y = BuildMatrix(neuralNetworkParameters.YDataSize);
            Matrix<double>[] Theta = null;
            if (!neuralNetworkParameters.IsThetaNull)
            {
                Theta = new Matrix<double>[neuralNetworkParameters.HiddenLayerLength + 1];

                for (int i = 0; i < (neuralNetworkParameters.HiddenLayerLength + 1); i++)
                {
                    Theta[i] = BuildMatrix(neuralNetworkParameters.ThetaSize[i]);
                }
            }

            return new NeuralNetwork
            {
                InputLayerSize = neuralNetworkParameters.InputLayerSize,
                HiddenLayerSize = neuralNetworkParameters.HiddenLayerSize,
                HiddenLayerLength = neuralNetworkParameters.HiddenLayerLength,
                OutputLayerSize = neuralNetworkParameters.OutputLayerSize,
                TrainingSize = neuralNetworkParameters.TrainingSize,
                Lambda = neuralNetworkParameters.Lambda,
                Epoch = neuralNetworkParameters.Epoch,
                X = X,
                y = y,
                Theta = Theta
            };
        }

        public void SendTheta(Matrix<double>[] theta)
        {

            var thetaSend = new double[theta.Length][,];

            for (int i = 0; i < theta.Length; i++)
            {
                thetaSend[i] = new double[theta[i].RowCount, theta[i].ColumnCount];
                for (int j = 0; j < theta[i].RowCount; j++)
                    for (int k = 0; k < theta[i].ColumnCount; k++)
                        thetaSend[i][j, k] = theta[i][j, k];
            }

            var thetaJson = JsonConvert.SerializeObject(thetaSend);

            CommunicationModule.SendData(thetaJson, true);

            /*
            if (P2P)
            {
                communicationTcp.SendData(thetaJson.Length.ToString());
                communicationTcp.SendDataSet(thetaJson);
            }
            else
            {
                CommunicationRabbitMqS2M.Publish(thetaJson);
            }
            */
        }

        public Matrix<double> BuildMatrix(int filesize)
        {
            double[][] data = ConvertStringToDataSet(CommunicationModule.ReceiveData(-1));

            /*
            if (P2P)
            {
                data = communicationTcp.ReceiveDataSet(filesize);
            }
            else
            {
                data = BuildDataSet(CommunicationRabbitMqM2s.Consume());
            }
            */
            return Matrix<double>.Build.Dense(data.Length, data[0].Length, (i, j) => data[i][j]);
        }

        public double[][] ConvertStringToDataSet(string str)
        {
            var lines = new List<double[]>();
            var Data = str.Split("\n");
            for (int i = 0; i < Data.Length; i++)
            {
                string[] line = Data[i].Split(',');
                var lineValues = line.Select(e => Convert.ToDouble(e)).ToArray();
                lines.Add(lineValues);
            }
            return lines.ToArray();
        }
    }
}