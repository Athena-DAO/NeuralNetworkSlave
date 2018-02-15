﻿using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    class NeuralNetworkParameters
    {
        public int InputLayerSize { get; set; }
        public int HiddenLayerSize { get; set; }
        public int HiddenLayerLength { get; set; }
        public int OutputLayerSize { get; set; }
        public double Lambda { get; set; }
        public int TrainingSize { get; set; }
        public int Epoch { get; set; }
        public int XDataSize { get; set; }
        public int YDataSize { get; set; }

    }
}
