using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
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
            NeuralNetworkCom neuralNetworkCom = JsonConvert.DeserializeObject<NeuralNetworkCom>(jsonCom);
            SendOk();
            var X = BuildMatrix(neuralNetworkCom.XDataSize);
            SendOk();
            var y = BuildMatrix(neuralNetworkCom.YDataSize);
            SendOk();
            return new NeuralNetwork
            {
                InputLayerSize = neuralNetworkCom.InputLayerSize,
                HiddenLayerSize = neuralNetworkCom.HiddenLayerSize,
                HiddenLayerLength = neuralNetworkCom.HiddenLayerLength,
                OutputLayerSize = neuralNetworkCom.OutputLayerSize,
                TrainingSize = neuralNetworkCom.TrainingSize,
                Lambda = neuralNetworkCom.Lambda,
                Epoch = neuralNetworkCom.Epoch,
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
            //Console.Write(stringBuilder.ToString());

            var temp = stringBuilder.ToString().Replace("\r", "");
            var Data = temp.Split("\n");
            Console.WriteLine("DataLength =" + Data.Length);
            for (int i = 0; i < Data.Length; i++)
            {
                string[] line = Data[i].Split(',');
                var lineValues = new double[line.Length];
                for (int j = 0; j < line.Length; j++)
                {
                    lineValues[j] = double.Parse(line[j]);
                }
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