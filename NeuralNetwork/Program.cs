using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;

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
        
        
        public static void Main(string[] args)
        {
            TcpListener tcpListener;
            var Ip = new Byte[4] { 127, 0, 0, 1 };

            Console.WriteLine("Enter The port number");
            int portNo = int.Parse(Console.ReadLine());
            tcpListener = new TcpListener(new IPAddress(Ip), portNo);
            tcpListener.Start();
            Socket socket = tcpListener.AcceptSocket();
            Console.WriteLine("Connected to  {0}", socket.RemoteEndPoint);
            try
            {
                Stream stream = new NetworkStream(socket);

                StreamReader streamReader = new StreamReader(stream);
                StreamWriter streamWriter = new StreamWriter(stream);
                string fileAppend = $"{DateTime.UtcNow:yyyyMMddHHmmssfff}";
                StreamWriter streamXValue = new StreamWriter(fileAppend + "_X_value" + socket.RemoteEndPoint.ToString().Split(":")[1]+".csv");
                StreamWriter streamYValue = new StreamWriter(fileAppend + "_Y_value" + socket.RemoteEndPoint.ToString().Split(":")[1] +".csv");
                streamWriter.AutoFlush = true;

                var neuralNetwork = BuildNeuralNetwork(streamReader,streamXValue,streamYValue);
                neuralNetwork.InitializeTheta();
                neuralNetwork.Train();
                streamWriter.WriteLine("Success");
                Console.Read();
            }
            catch (Exception E)
            {
                Console.WriteLine(E);
                var x = Console.ReadLine();
            }
            finally
            {
                socket.Close();
            }
        }
        
 
        public static NeuralNetwork BuildNeuralNetwork(StreamReader streamReader,StreamWriter streamXValue , StreamWriter streamYValue)
        {



            int InputLayerSize = int.Parse(streamReader.ReadLine());
            int HiddenLayerSize = int.Parse(streamReader.ReadLine());
            int HiddenLayerLength = int.Parse(streamReader.ReadLine());
            int OutputLayerSize = int.Parse(streamReader.ReadLine());
            int TrainingSize = int.Parse(streamReader.ReadLine());
            double Lambda = double.Parse(streamReader.ReadLine());
            int Epoch = int.Parse(streamReader.ReadLine());
            var  X = BuildMatrix(streamReader,streamXValue);
            var y = BuildMatrix(streamReader,streamYValue);

            return new NeuralNetwork
            {
                InputLayerSize = InputLayerSize,
                HiddenLayerSize = HiddenLayerSize,
                HiddenLayerLength = HiddenLayerLength,
                OutputLayerSize = OutputLayerSize,
                TrainingSize = TrainingSize,
                Lambda = Lambda,
                Epoch = Epoch,
                X = X,
                y = y
            };

        }

        public static Matrix<double> BuildMatrix(StreamReader streamReader,StreamWriter streamWrite)
        {

            int filesize = int.Parse(streamReader.ReadLine());
            char[] buffer = new char[1024];
            var lines = new List<double[]>();
            int size = 0;
            int num;
            StringBuilder stringBuilder = new StringBuilder(filesize);
            while (size < (filesize) && (num = streamReader.Read(buffer, 0, buffer.Length)) != 0)
            {
                streamWrite.Write(new String(buffer, 0, num));
                stringBuilder.Append(buffer, 0, num);
                size += num;
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

      /*
        private static void Main(string[] args)
        {
            var Theta1 = ReadCsv("Theta1_value.csv");
            var Theta2 = ReadCsv("Theta2_value.csv");
            var X = ReadCsv("X_value.csv");
            var y = ReadCsv("Y_value.csv");
            var x_Test = ReadCsv("X_predict.csv");

            NeuralNetwork neuralNetwork = new NeuralNetwork()
            {
                X = X,
                y = y,
                InputLayerSize = 400,
                HiddenLayerSize = 25,
                HiddenLayerLength = 1,
                OutputLayerSize = 10,
                TrainingSize = 5000,
                Lambda = 3,
                Epoch = 50
            };
            neuralNetwork.InitializeTheta();
            //neuralNetwork.ReadParams(Theta, X, y);
            neuralNetwork.Train();

            double[] predictions = neuralNetwork.predict(x_Test);

            WriteCsv("TrainedTheta1.csv", neuralNetwork.Theta[0]);
            WriteCsv("TrainedTheta2.csv", neuralNetwork.Theta[1]);
            Console.ReadLine();
        }
        */
    }
}