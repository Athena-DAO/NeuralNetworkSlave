using NeuralNetwork.Communication;
using NeuralNetwork.Models;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Threading;

namespace NeuralNetwork.Logging
{
    internal class LogService
    {
        private Object LogLock;
        
        public CommunicationModule communicationModule { get; set; }
        public List<Log> Logs { get; set; }
        private bool stopLogs = false;

        public LogService()
        {
            LogLock = new object();
            Logs = new List<Log>();
        }

        public void AddLog(string logType, string message)
        {
            lock (LogLock)
            {
                Logs.Add(new Log() { LogType = logType, Message = message });
            }
        }

        public void Service()
        {
            while (!stopLogs)
            {
                SendLogs();
                Thread.Sleep(5000);
            }
        }

        public void SendLogs()
        {

                lock (LogLock)
                {
                    if (Logs.Count > 0)
                    {
                        communicationModule.SendData(JsonConvert.SerializeObject(Logs), false);
                        Logs.Clear();
                    }
                }
            
        }

        public void StartLogService()
        {
            (new Thread(Service)).Start();
            
        }

        public void StopLogService()
        {
            SendLogs();
            stopLogs = true;
        }
    }
}