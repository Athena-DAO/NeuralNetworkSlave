using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;

namespace NeuralNetwork
{
    internal class CommunicationLayer
    {
        private TcpListener tcpListener;
        private Socket socket;
        private Stream stream;

        public CommunicationLayer(int portNo)
        {
            var Ip = new Byte[4] { 127, 0, 0, 1 };
            try
            {
                tcpListener = new TcpListener(new IPAddress(Ip), portNo);
                tcpListener.Start();                
            }
            catch(Exception e)
            {
                Console.WriteLine("Exception ", e);
            }
        }

        public void AcceptConnection()
        {
            socket = tcpListener.AcceptSocket();
            Console.WriteLine("Connected to  {0}", socket.RemoteEndPoint);
            stream = new NetworkStream(socket);
        }
        public NeuralNetwork BuildNeuralNetwork()
        {
            var bytes = new byte[1024];
            int received = stream.Read(bytes, 0, 1024);
            string jsonCom = Encoding.ASCII.GetString(bytes, 0, received);
            NeuralNetworkParameters neuralNetworkParameters = JsonConvert.DeserializeObject<NeuralNetworkParameters>(jsonCom);
            SendOk();
            var X = BuildMatrix(neuralNetworkParameters.XDataSize);
            SendOk();
            var y = BuildMatrix(neuralNetworkParameters.YDataSize);
            SendOk();
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
            var buffer = new byte[1024];
            var lines = new List<double[]>();
            int receivedSize = 0;
            int bytesReceived;
            StringBuilder stringBuilder = new StringBuilder(filesize);

            while (receivedSize < (filesize) && (bytesReceived = stream.Read(buffer, 0, 1024)) != 0)
            {
                String msg = Encoding.ASCII.GetString(buffer, 0, bytesReceived);
                stringBuilder.Append(msg, 0, bytesReceived);
                receivedSize += bytesReceived;
            }
            var Data = stringBuilder.ToString().Split("\n");
            for (int i = 0; i < Data.Length; i++)
            {
                string[] line = Data[i].Split(',');
                var lineValues = line.Select(e => Convert.ToDouble(e)).ToArray();
                lines.Add(lineValues);
            }
            var data = lines.ToArray();
            return Matrix<double>.Build.Dense(data.Length, data[0].Length, (i, j) => data[i][j]);
        }

        public void Close()
        {
            socket.Close();
        }

        private void SendOk()
        {
            var bytes = Encoding.ASCII.GetBytes("Ok");
            stream.Write(bytes, 0, bytes.Length);
        }
    }
}