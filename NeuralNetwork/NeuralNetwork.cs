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
        public double Lambda { get; set; }

        //public Matrix<double>[] Theta { get; set; }
        //public Matrix<double> X { get; set; }
        //public Matrix<double> y { get; set; }
        
        
        Matrix<double> []Theta;
        Matrix<double> X;
        Matrix<double> y;
        Matrix<double> []Activation;
        Matrix<double> []ActivationWithBias;
        Matrix<double> []Z;

        public NeuralNetwork()
        {
            //Theta = new Matrix<double>[HiddenLayerLength + 1];
            //Theta[0] = Matrix<double>.Build.Random(HiddenLayerSize, InputLayerSize + 1);
            //Theta[HiddenLayerLength] = Matrix<double>.Build.Random(OutputLayerSize, HiddenLayerSize + 1);
            //for (int i = 1; i < HiddenLayerLength; i++)
            //    Theta[i] = Matrix<double>.Build.Random(HiddenLayerSize + 1, HiddenLayerSize);
           
            //Activation = new Matrix<double>[HiddenLayerSize+2];
            //ActivationWithBias = new Matrix<double>[HiddenLayerSize + 2];
            //Z = new Matrix<double>[HiddenLayerSize + 2];

            //X.CopyTo(Activation[0]); 
        }

        public void ReadParams(Matrix<double>[] Theta , Matrix<double> X, Matrix<double> y)
        {
            this.Theta = Theta;
            this.X = X;
            this.y = y;

        }

        public double Cost()
        {
            Activation = new Matrix<double>[HiddenLayerLength + 2];
            ActivationWithBias = new Matrix<double>[HiddenLayerLength + 2];
            Z = new Matrix<double>[HiddenLayerLength + 2];

            Activation[0] = Matrix<double>.Build.Dense(X.RowCount, X.ColumnCount, (i, j) => X[i, j]);
//            X.CopyTo(Activation[0]);
            

            double cost;

            Matrix<double> Y = Matrix<double>.Build.Dense(TrainingSize, OutputLayerSize, (i, j) => ( y[i, 0] == j ? 1 : 0 ));

            for (int i = 0; i < HiddenLayerLength+1; i++)
            {
                ActivationWithBias[i] = Matrix<double>.Build.Dense(Activation[i].RowCount, Activation[i].ColumnCount + 1, (x, y) => (y == 0 ? 1 : Activation[i][x, y - 1]));
                Z[i+1] = ActivationWithBias[i] * Theta[i].Transpose();
                Activation[i+1] = Sigmoid(Z[i+1]);
            }

            //J=(1.0/m)*sum(sum(-Y.*log(a3)-(1-Y).*log(1-a3)),2) + reg;
            var lg1 = Activation[HiddenLayerLength + 1].PointwiseLog();
            var temp2 = -Y.PointwiseMultiply(lg1);
            var a = (-Y).PointwiseMultiply(Activation[HiddenLayerLength + 1].PointwiseLog());
            var t = a.RowSums().Sum();
           // var b = (1 - Y).PointwiseMultiply( (1-Activation[HiddenLayerLength + 1]).PointwiseLog() );

            var b  =  (1 - Y).PointwiseMultiply( Activation[HiddenLayerLength +1].Map( m => (1-m) ).PointwiseLog() );   

            //reg = lambda/(2.0*m)*(sum(sum(Theta1(:,2:size(Theta1,2)).^2),2) + sum(sum(Theta2(:,2:size(Theta2,2)).^2),2));


            

            double sum=0.0;
            for (int i = 0; i < HiddenLayerLength + 1; i++)
                 sum += Theta[i].SubMatrix(0, Theta[i].RowCount, 1, Theta[i].ColumnCount - 1).Map(m => m*m)
                    .ColumnSums().Sum();
            
            double regularization = Lambda / (2.0 * TrainingSize) * sum;


            cost = (1.0 / TrainingSize) * (a - b).ColumnSums().Sum() + regularization;



            return cost;

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
