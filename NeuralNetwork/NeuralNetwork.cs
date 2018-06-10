using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Logging;
using System;
using System.Threading;
using Newtonsoft.Json;
using NeuralNetwork.Models;
using Microsoft.Extensions.Configuration;

namespace NeuralNetwork
{
    internal class NeuralNetwork
    {
        public int InputLayerSize { get; set; }
        public int HiddenLayerSize { get; set; }
        public int HiddenLayerLength { get; set; }
        public int OutputLayerSize { get; set; }
        public int TrainingSize { get; set; }
        public double Lambda { get; set; }
        public int Epoch { get; set; }


        public Matrix<double>[] Theta { get; set; }
        public Matrix<double> X { get; set; }
        public Matrix<double> y { get; set; }

        public LogService LogService { get; set; }
        public IConfiguration Configuration { get; set; }

        private alglib.minlbfgsstate state;
        private double cost;
        private const int SLEEP = 1000;
        
        public NeuralNetwork()
        {
        }

        public void InitializeTheta()
        {
            Theta = new Matrix<double>[HiddenLayerLength + 1];
            Theta[0] = Matrix<double>.Build.Random(HiddenLayerSize, InputLayerSize + 1);
            Theta[HiddenLayerLength] = Matrix<double>.Build.Random(OutputLayerSize, HiddenLayerSize + 1);
            for (int i = 1; i < HiddenLayerLength; i++)
                Theta[i] = Matrix<double>.Build.Random(HiddenLayerSize, HiddenLayerSize+1);
        }
        public void ReadParams(Matrix<double>[] Theta, Matrix<double> X, Matrix<double> y)
        {
            this.Theta = Theta;
            this.X = X;
            this.y = y;
        }

        public double Cost()
        {
            double cost;
            double regularization;
            double regularizationSum = 0.0;

            Matrix<double>[] Activation;
            Matrix<double>[] ActivationWithBias;
            Matrix<double>[] Z;

            //Initialization of Matrix Array
            Activation = new Matrix<double>[HiddenLayerLength + 2];
            ActivationWithBias = new Matrix<double>[HiddenLayerLength + 1];
            Z = new Matrix<double>[HiddenLayerLength + 2];

            //Initialization of Matrix Activation[0] and Y
            Activation[0] = Matrix<double>.Build.Dense(X.RowCount, X.ColumnCount, (i, j) => X[i, j]);
            Matrix<double> Y = Matrix<double>.Build.Dense(TrainingSize, OutputLayerSize, (i, j) => (y[i, 0] == j ? 1 : 0));

            for (int i = 0; i < HiddenLayerLength + 1; i++)
            {
                ActivationWithBias[i] = Matrix<double>.Build.Dense(Activation[i].RowCount, Activation[i].ColumnCount + 1, (x, y) => (y == 0 ? 1 : Activation[i][x, y - 1]));
                Z[i + 1] = ActivationWithBias[i] * Theta[i].Transpose();
                Activation[i + 1] = Sigmoid(Z[i + 1]);
            }

            // Calcualting Regularization
            for (int i = 0; i < HiddenLayerLength + 1; i++)
                regularizationSum += Theta[i].SubMatrix(0, Theta[i].RowCount, 1, Theta[i].ColumnCount - 1).Map(m => m * m)
                   .ColumnSums().Sum();
            regularization = Lambda / (2.0 * TrainingSize) * regularizationSum;

            //Calculating cost
            cost = (1.0 / TrainingSize) * (
                (-Y).PointwiseMultiply(Activation[HiddenLayerLength + 1].PointwiseLog()) +
                (Y - 1).PointwiseMultiply(Activation[HiddenLayerLength + 1].Map(m => (1 - m)).PointwiseLog())
                ).ColumnSums().Sum() + regularization;
            this.cost = cost;
            return cost;
        }

        public void Cost(double[] thetaUnpacked, ref double cost, double[] grad, object obj)
        {
            Matrix<double>[] Theta = PackTheta(thetaUnpacked);
            Matrix<double>[] Activation;
            Matrix<double>[] ActivationWithBias;
            Matrix<double>[] Z;
            Matrix<double>[] ThetaGradient;
            Matrix<double>[] ThetaWithoutBias;
            Matrix<double>[] Delta;

            double regularization;
            double regularizationSum = 0.0;

            //Initialization of Matrix Array
            Activation = new Matrix<double>[HiddenLayerLength + 2];
            ActivationWithBias = new Matrix<double>[HiddenLayerLength + 1];
            Z = new Matrix<double>[HiddenLayerLength + 2];
            ThetaGradient = new Matrix<double>[HiddenLayerLength + 1];
            ThetaWithoutBias = new Matrix<double>[HiddenLayerLength + 1];
            Delta = new Matrix<double>[HiddenLayerLength + 1];

            //Initialization of Matrix Activation[0] and Y
            Activation[0] = Matrix<double>.Build.Dense(X.RowCount, X.ColumnCount, (i, j) => X[i, j]);
            Matrix<double> Y = Matrix<double>.Build.Dense(TrainingSize, OutputLayerSize, (i, j) => (y[i, 0] == j ? 1 : 0));

            for (int i = 0; i < HiddenLayerLength + 1; i++)
            {
                ActivationWithBias[i] = Matrix<double>.Build.Dense(Activation[i].RowCount, Activation[i].ColumnCount + 1, (x, y) => (y == 0 ? 1 : Activation[i][x, y - 1]));
                Z[i + 1] = ActivationWithBias[i] * Theta[i].Transpose();
                Activation[i + 1] = Sigmoid(Z[i + 1]);
            }

            // Calcualting Regularization
            for (int i = 0; i < HiddenLayerLength + 1; i++)
                regularizationSum += Theta[i].SubMatrix(0, Theta[i].RowCount, 1, Theta[i].ColumnCount - 1).Map(m => m * m)
                   .ColumnSums().Sum();
            regularization = Lambda / (2.0 * TrainingSize) * regularizationSum;

            //Calculating cost
            cost = (1.0 / TrainingSize) * (
                (-Y).PointwiseMultiply(Activation[HiddenLayerLength + 1].PointwiseLog()) +
                (Y - 1).PointwiseMultiply(Activation[HiddenLayerLength + 1].Map(m => (1 - m)).PointwiseLog())
                ).ColumnSums().Sum() + regularization;
            this.cost = cost;

            //Calculating gradient at the output layer
            Delta[HiddenLayerLength] = Activation[HiddenLayerLength + 1] - Y;
            ThetaWithoutBias[HiddenLayerLength] = Matrix<double>.Build.Dense(Theta[HiddenLayerLength].RowCount, Theta[HiddenLayerLength].ColumnCount,
                                                    (x, y) => (y == 0 ? 0 : Theta[HiddenLayerLength][x, y]));
            ThetaGradient[HiddenLayerLength] = (1.0 / TrainingSize) * (Delta[HiddenLayerLength].Transpose() * ActivationWithBias[HiddenLayerLength]) +
                                Lambda / TrainingSize * ThetaWithoutBias[HiddenLayerLength];

            //Calculating gradient at the hidden Layers
            for (int i = HiddenLayerLength - 1; i >= 0; i--)
            {
                ThetaWithoutBias[i] = Matrix<double>.Build.Dense(Theta[i].RowCount, Theta[i].ColumnCount, (x, y) => (y == 0 ? 0 : Theta[i][x, y]));
                Delta[i] = (Delta[i + 1] * Theta[i + 1].SubMatrix(0, Theta[i + 1].RowCount, 1, Theta[i + 1].ColumnCount - 1)).
                        PointwiseMultiply(SigmoidGradient(Z[1]));
                ThetaGradient[i] = (1.0 / TrainingSize) * (Delta[i].Transpose() * ActivationWithBias[i]) +
                                Lambda / TrainingSize * ThetaWithoutBias[i];
            }

            UnpackTheta(ThetaGradient, ref grad);
        }

        public int Train()
        {
            if (Theta == null)
                InitializeTheta();

            double epsg = 0.0000000001;
            double epsf = 0;
            double epsx = 0;
            var thetaUnpack = UnpackTheta(Theta);

            alglib.minlbfgsreport rep;
            alglib.minlbfgscreate(1, thetaUnpack, out state);
            alglib.minlbfgssetcond(state, epsg, epsf, epsx, Epoch);
            Thread logThread = new Thread(LogCost);
            Cost();
            logThread.Start();
            alglib.minlbfgsoptimize(state, Cost, null, null);
            alglib.minlbfgsresults(state, out thetaUnpack, out rep);
            Console.WriteLine("Termination type {0} Iteration Count {1}", rep.terminationtype, rep.iterationscount);
            var theta = PackTheta(thetaUnpack);
            this.Theta = theta;

            while(logThread.IsAlive)
            {
                Thread.Sleep(SLEEP);
            }
            return rep.terminationtype;
        }

        public void LogCost()
        {
            double[] thetaUnpack;
            alglib.minlbfgsreport report = new alglib.minlbfgsreport();
            do
            {
                alglib.minlbfgsresults(state, out thetaUnpack, out report);
                LogService.AddLog("info", JsonConvert.SerializeObject(
                    new InfoLog() {
                    Iteration = report.iterationscount,
                    Cost= cost}));
                Console.WriteLine("Iteration {0}| Cost {1}", report.iterationscount, cost);
                Thread.Sleep(int.Parse($"{Configuration["log-generation-time-interval"]}"));
            }
            while (report.iterationscount != (Epoch));
        }

        public double[] UnpackTheta(Matrix<double>[] thetaPacked)
        {
            int sum = 0, k = 0;
            double[] ThetaUnpack;

            sum = HiddenLayerSize * (InputLayerSize + 1) + (HiddenLayerLength - 1) * (HiddenLayerSize) * (HiddenLayerSize + 1) + OutputLayerSize * (HiddenLayerSize + 1);
            ThetaUnpack = new double[sum];

            foreach (var theta in thetaPacked)
            {
                for (int i = 0; i < theta.RowCount; i++)
                    for (int j = 0; j < theta.ColumnCount; j++)
                        ThetaUnpack[k++] = theta[i, j];
            }

            return ThetaUnpack;
        }

        public void UnpackTheta(Matrix<double>[] thetaPacked, ref double[] ThetaUnpack)
        {
            int sum = 0, k = 0;

            sum = HiddenLayerSize * (InputLayerSize + 1) + (HiddenLayerLength - 1) * (HiddenLayerSize) * (HiddenLayerSize + 1) + OutputLayerSize * (HiddenLayerSize + 1);

            foreach (var theta in thetaPacked)
            {
                for (int i = 0; i < theta.RowCount; i++)
                    for (int j = 0; j < theta.ColumnCount; j++)
                        ThetaUnpack[k++] = theta[i, j];
            }
        }

        public Matrix<double>[] PackTheta(double[] thetaUnpack)
        {
            int k = 0;
            Matrix<double>[] thetaPack = new Matrix<double>[HiddenLayerLength + 1];

            //Input Layer
            thetaPack[0] = Matrix<double>.Build.Dense(HiddenLayerSize, InputLayerSize + 1, (i, j) => (thetaUnpack[k + (InputLayerSize + 1) * i + j]));
            k += HiddenLayerSize * (InputLayerSize + 1);

            //Hidden Layers
            for (int i = 1; i < HiddenLayerLength; i++)
            {
                thetaPack[i] = Matrix<double>.Build.Dense(HiddenLayerSize, HiddenLayerSize + 1, (x, y) => (thetaUnpack[k + (HiddenLayerSize + 1) * x + y]));
                k += (HiddenLayerSize) * (HiddenLayerSize + 1);
            }

            //Output Layer
            thetaPack[HiddenLayerLength] = Matrix<double>.Build.Dense(OutputLayerSize, HiddenLayerSize + 1, (i, j) => (thetaUnpack[k + (HiddenLayerSize + 1) * i + j]));

            return thetaPack;
        }

        public double[] Predict(Matrix<double> X)
        {
            Matrix<double>[] Activation;
            Matrix<double>[] ActivationWithBias;
            Matrix<double>[] Z;
            var predictions = new double[X.RowCount];

            //Initialization of Matrix Array
            Activation = new Matrix<double>[HiddenLayerLength + 2];
            ActivationWithBias = new Matrix<double>[HiddenLayerLength + 1];
            Z = new Matrix<double>[HiddenLayerLength + 2];

            //Initialization of Matrix Activation[0] and Y
            Activation[0] = Matrix<double>.Build.Dense(X.RowCount, X.ColumnCount, (i, j) => X[i, j]);

            for (int i = 0; i < HiddenLayerLength + 1; i++)
            {
                ActivationWithBias[i] = Matrix<double>.Build.Dense(Activation[i].RowCount, Activation[i].ColumnCount + 1, (x, y) => (y == 0 ? 1 : Activation[i][x, y - 1]));
                Z[i + 1] = ActivationWithBias[i] * Theta[i].Transpose();
                Activation[i + 1] = Sigmoid(Z[i + 1]);
            }

            for (int i = 0; i < Activation[HiddenLayerLength + 1].RowCount; i++)
            {
                double max = 0, index = -1;
                for (int j = 0; j < Activation[HiddenLayerLength + 1].ColumnCount; j++)
                    if (Activation[HiddenLayerLength + 1][i, j] > max)
                    {
                        max = Activation[HiddenLayerLength + 1][i, j];
                        index = j;
                    }
                predictions[i] = index;
            }
            return predictions;
        }

        public Matrix<double> Sigmoid(Matrix<double> matrix)
        {
            return matrix.Map(m => 1 / (1 + Math.Exp(-m)));
        }

        public Matrix<double> SigmoidGradient(Matrix<double> matrix)
        {
            return Sigmoid(matrix).PointwiseMultiply(1 - Sigmoid(matrix));
        }
    }
}