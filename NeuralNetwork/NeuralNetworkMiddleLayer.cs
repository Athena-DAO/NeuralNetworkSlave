using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;
namespace NeuralNetwork
{
    class NeuralNetworkMiddleLayer
    {

        private CommunicationLayer communicationLayer;

        public NeuralNetworkMiddleLayer(CommunicationLayer communicationLayer)
        {
            this.communicationLayer = communicationLayer;
        }


        public NeuralNetwork BuildNeuralNetwork()
        {
            var json = communicationLayer.ReceiveData();
            NeuralNetworkParameters neuralNetworkParameters = JsonConvert.DeserializeObject<NeuralNetworkParameters>(json);

             var X = BuildMatrix(neuralNetworkParameters.XDataSize);
             var y = BuildMatrix(neuralNetworkParameters.YDataSize);
            Matrix<double>[] Theta = null;
            if (!neuralNetworkParameters.IsThetaNull)
            {
                 Theta = new Matrix<double>[neuralNetworkParameters.HiddenLayerLength + 1];

                for(int i=0;i<(neuralNetworkParameters.HiddenLayerLength+1);i++)
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

           
           for(int i=0;i<theta.Length;i++)
           {
                thetaSend[i] = new double[theta[i].RowCount, theta[i].ColumnCount];
                for (int j = 0; j < theta[i].RowCount; j++)
                    for (int k = 0; k < theta[i].ColumnCount; k++)
                        thetaSend[i][j, k] = theta[i][j, k];
           }

            var thetaJson = JsonConvert.SerializeObject(thetaSend);
            communicationLayer.SendData(thetaJson.Length.ToString());
            communicationLayer.SendDataSet(thetaJson);

        }

        public Matrix<double> BuildMatrix(int filesize)
        {
            var data = communicationLayer.ReceiveDataSet(filesize);
            return Matrix<double>.Build.Dense(data.Length, data[0].Length, (i, j) => data[i][j]);
        }
    }
}
