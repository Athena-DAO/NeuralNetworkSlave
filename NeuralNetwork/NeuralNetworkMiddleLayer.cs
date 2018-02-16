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
            var json = communicationLayer.GetJSONData();
            NeuralNetworkParameters neuralNetworkParameters = JsonConvert.DeserializeObject<NeuralNetworkParameters>(json);

             var X = BuildMatrix(neuralNetworkParameters.XDataSize);
             var y = BuildMatrix(neuralNetworkParameters.YDataSize);

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
                y = y
            };
        }

        public Matrix<double> BuildMatrix(int filesize)
        {
            var data = communicationLayer.GetDataSet(filesize);
            return Matrix<double>.Build.Dense(data.Length, data[0].Length, (i, j) => data[i][j]);
        }
    }
}
