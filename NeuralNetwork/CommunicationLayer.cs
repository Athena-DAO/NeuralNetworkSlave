using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;

namespace NeuralNetwork
{
    class CommunicationsLayer
    {
        public CommunicationModule server { get; set; }
        public string PipelineId { get; set; }
        public CommunicationsLayer()
        {
            server = new CommunicationModule("13.232.7.234", 6000);
        }

        public void SendCommunicationServerParameters()
        {
            server.SendData(JsonConvert.SerializeObject(new CommunicationServerParameters()
            {
                PipelineId = PipelineId,
                IsMaster = false
            }));

        }

        public IPEndPoint GetPeerIPEndPoint()
        {
            return GetIpEndPoint(server.ReceiveData());
        }


        public IPEndPoint GetIpEndPoint(string ipEndPointString)
        {
            var localEndPointList = ipEndPointString.Split(':');
            var ipAddress = localEndPointList[0].Split('.').Select(i => Convert.ToByte(i)).ToArray();
            return new IPEndPoint(new IPAddress(ipAddress), int.Parse(localEndPointList[1]));
        }
    }
}
