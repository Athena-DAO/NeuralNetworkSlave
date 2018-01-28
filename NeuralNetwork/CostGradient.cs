using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
namespace NeuralNetwork
{
    class CostGradient
    {
        public double cost { get; set; }
        public double[] ThetaGradients { get; set; }

        public CostGradient(double cost,Matrix<double>[] ThetaGradient)
        {
            this.cost = cost;
            int sum = 0, k = 0;
            double[] thetaGradients;
            foreach (var theta in ThetaGradient)
            {
                sum += theta.RowCount * theta.ColumnCount;
            }
            thetaGradients = new double[sum];

            foreach (var theta in ThetaGradient)
            {
                for (int i = 0; i < theta.RowCount; i++)
                    for (int j = 0; i < theta.ColumnCount; j++)
                        thetaGradients[k++] = theta[i, j];
            }
            ThetaGradients=thetaGradients;
        }

        public CostGradient()
        {

        }
    }
}
