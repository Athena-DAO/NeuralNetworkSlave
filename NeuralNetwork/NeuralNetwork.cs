using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;

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

        public CostGradient Cost()
        {

            Matrix<double>[] Activation;
            Matrix<double>[] ActivationWithBias;
            Matrix<double>[] Z;
            Matrix<double>[] ThetaGradient;
            Matrix<double>[] ThetaWithoutBias;
            Matrix<double>[] Delta;

            double cost;
            double regularization;
            double regularizationSum = 0.0;

            //Initialization of Matrix Array
            Activation = new Matrix<double>[HiddenLayerLength + 2];
            ActivationWithBias = new Matrix<double>[HiddenLayerLength + 2];
            Z = new Matrix<double>[HiddenLayerLength + 2];
            ThetaGradient = new Matrix<double>[HiddenLayerLength + 2];
            ThetaWithoutBias = new Matrix<double>[HiddenLayerLength + 2];
            Delta = new Matrix<double>[HiddenLayerLength + 2];

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



            //Calculating gradient at the output layer
            Delta[HiddenLayerLength] = Activation[HiddenLayerLength + 1] - Y;

            ThetaWithoutBias[HiddenLayerLength] = Matrix<double>.Build.Dense(Theta[HiddenLayerLength].RowCount, Theta[HiddenLayerLength].ColumnCount, 
                                                    (x, y) => (y == 0 ? 0 : Theta[HiddenLayerLength][x, y]));

            ThetaGradient[HiddenLayerLength] = (1.0 / TrainingSize) * (Delta[HiddenLayerLength].Transpose() * ActivationWithBias[HiddenLayerLength]) +
                                Lambda / TrainingSize * ThetaWithoutBias[HiddenLayerLength];

            //Calculating gradient at the hidden Layers
            for (int i= HiddenLayerLength-1; i>=0; i--)
            {
                ThetaWithoutBias[i] = Matrix<double>.Build.Dense(Theta[i].RowCount, Theta[i].ColumnCount, (x, y) => (y == 0 ? 0 : Theta[i][x, y]));
                Delta[i] = (Delta[i+1] * Theta[i+1].SubMatrix(0, Theta[i+1].RowCount, 1, Theta[i+1].ColumnCount - 1)).
                        PointwiseMultiply(SigmoidGradient(Z[1]));
                ThetaGradient[i] = (1.0 / TrainingSize) * (Delta[i].Transpose() * ActivationWithBias[i]) +
                                Lambda / TrainingSize * ThetaWithoutBias[i];
            }

            return new CostGradient(cost, ThetaGradient);
        }

        public void Train()
        {



            var thetaUnpack = UnpackTheta();


            double epsg = 0.0000000001;
            double epsf = 0;
            double epsx = 0;
            int maxits = 400;



            alglib.minlbfgsstate state;
            alglib.minlbfgsreport rep;

            alglib.minlbfgscreate(1,thetaUnpack, out state);
            alglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits);
            alglib.minlbfgsoptimize(state, CostGradient, null, null);
            alglib.minlbfgsresults(state, out x, out rep);

        }


        private double[] UnpackTheta()
        {
            int sum=0,k=0;
            double[] ThetaUnpack;
            foreach (var theta in this.Theta)
            {
                sum += theta.RowCount * theta.ColumnCount;
            }
            ThetaUnpack = new double[sum];

            foreach (var theta in this.Theta)
            {
                for (int i = 0; i < theta.RowCount; i++)
                    for (int j = 0; i < theta.ColumnCount; j++)
                        ThetaUnpack[k++] = theta[i, j];
            }
            return ThetaUnpack;
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