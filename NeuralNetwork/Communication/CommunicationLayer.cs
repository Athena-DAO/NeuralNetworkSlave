using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using RabbitMQ.Client;
using NeuralNetwork.Communication;
using NeuralNetwork.Model;
using Microsoft.Extensions.Configuration;
using System.IO;

namespace NeuralNetwork
{
    class CommunicationsLayer
    {
        public CommunicationModule server { get; set; }
        public string PipelineId { get; set; }
        public IConfiguration Configuration { get; set; }

        public CommunicationsLayer()
        {
            BuildConfiguration();
            server = new CommunicationModule($"{Configuration["Ip-CommunicationServer"]}", int.Parse($"{Configuration["Port-CommunicationServer"]}"));
        }

        public void BuildConfiguration()
        {
            var builder = new ConfigurationBuilder().SetBasePath(Directory.GetCurrentDirectory()).AddJsonFile("appsettings.json");
            Configuration = builder.Build();
        }
        public CommunicationResponse GetCommunicationResonse()
        {
            return JsonConvert.DeserializeObject<CommunicationResponse>(server.ReceiveData());
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
