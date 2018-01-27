using MathNet.Numerics.LinearAlgebra;
using System;

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

        private Matrix<double>[] Theta;
        private Matrix<double> X;
        private Matrix<double> y;

        public NeuralNetwork()
        {
        }

        public void ReadParams(Matrix<double>[] Theta, Matrix<double> X, Matrix<double> y)
        {
            this.Theta = Theta;
            this.X = X;
            this.y = y;
        }

        public double Cost()
        {
            Matrix<double>[] Activation;
            Matrix<double>[] ActivationWithBias;
            Matrix<double>[] Z;
            double cost;
            double regularization;
            double regularizationSum = 0.0;

            //Initialization of Matrix Array
            Activation = new Matrix<double>[HiddenLayerLength + 2];
            ActivationWithBias = new Matrix<double>[HiddenLayerLength + 2];
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

            return cost;
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