using NeuralNetwork.Models;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Threading;

namespace NeuralNetwork.Logging
{
    internal class LoggingService
    {
        private Object LogLock;
        public Object communicationModuleLock { get; set; }
        public CommunicationModule communicationModule { get; set; }
        public List<Log> Logs { get; set; }
        private bool stopLogs=false;

        public LoggingService()
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

        public void SendLogs()
        {
           
            while (!stopLogs)
            {
                lock (communicationModuleLock)
                {
                    lock (LogLock)
                    {
                        communicationModule.SendData(JsonConvert.SerializeObject(Logs));
                        Logs.Clear();
                    }
                }
                Thread.Sleep(5000);
            }
        }

        public void StartLogsService()
        {
            var t = new Thread(SendLogs);
            t.Start();
        }

        public void StopLogsService()
        {
            stopLogs = true;
        }
    }
}