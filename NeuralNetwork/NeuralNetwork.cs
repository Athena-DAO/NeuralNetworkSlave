using System;
using System.Collections.Generic;
using System.Text;
using MathNet;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork
{
    class NeuralNetwork
    {

        public int InputLayerSize { get; set; }
        public int HiddenLayerSize { get; set; }
        public int HiddenLayerLength { get; set; }
        public int OutputLayerSize { get; set; }
        public int TrainingSize { get; set; }
        public int MyProperty { get; set; }
        public double Lambda { get; set; }

        //double [][,] Theta_t;
        //double[,] X;
        //double[,] y;

        Matrix<double> []Theta;
        Matrix<double> X;
        Matrix<double> y;
        Matrix<double> []Activation;
        Matrix<double> []ActivationWithBias;
        Matrix<double> []Z;

        public NeuralNetwork()
        {
            Theta = new Matrix<double>[HiddenLayerLength + 1];
            Theta[0] = Matrix<double>.Build.Random(HiddenLayerSize, InputLayerSize + 1);
            Theta[HiddenLayerLength] = Matrix<double>.Build.Random(OutputLayerSize, HiddenLayerSize + 1);
            for (int i = 1; i < HiddenLayerLength; i++)
                Theta[i] = Matrix<double>.Build.Random(HiddenLayerSize + 1, HiddenLayerSize);
           
            Activation = new Matrix<double>[HiddenLayerSize+2];
            ActivationWithBias = new Matrix<double>[HiddenLayerSize + 2];
            Z = new Matrix<double>[HiddenLayerSize + 2];

            X.CopyTo(Activation[0]); 
        }

        public double Cost()
        {
            

            for (int i = 0; i < HiddenLayerSize; i++)
            {
                ActivationWithBias[0] = Matrix<double>.Build.Dense(Activation[0].RowCount, Activation[0].ColumnCount + 1, (x, y) => (y == 0 ? 0 : X[x, y - 1]));
                Z[1] = Activation[0] * Theta[0].Transpose();
                Activation[1] = Sigmoid(Z[1]);
            }
            

        }



        public Matrix<double> Sigmoid(Matrix<double> matrix)
        {
            return matrix.Map(m => 1/(1+Math.Exp(-m)));
        }

        public Matrix<double> SigmoidGradient(Matrix<double> matrix)
        {
            return Sigmoid(matrix).PointwiseMultiply(1 - Sigmoid(matrix));          
        }
    }
}
