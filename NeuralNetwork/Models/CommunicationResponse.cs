using System;
using System.Collections.Generic;
using System.Net;
using System.Text;

namespace NeuralNetwork.Model
{
    class CommunicationResponse
    {
        public string EndPoint { get; set; }
        public int QueueNumber { get; set; }
        public bool P2P { get; set; }
    }
}
