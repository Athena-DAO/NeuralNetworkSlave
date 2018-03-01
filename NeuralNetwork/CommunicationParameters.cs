using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    class CommunicationParameters
    {
        public string Id { get; set; }
        public int Port { get; set; }
        public string MasterIp { get; set; }
        public string MasterId { get; set; }
    }
}
