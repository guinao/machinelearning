using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TimeSeries;

namespace SrAnomalyDetection
{
    public class Program
    {
        private sealed class TimeSeriesData
        {
            public float Value;

            public TimeSeriesData(float value)
            {
                Value = value;
            }
        }

        private sealed class TimeSeriesDataDouble
        {
            [LoadColumn(0)]
            public double Value { get; set; }
        }

        private sealed class SrCnnAnomalyDetection
        {
            [VectorType]
            public double[] Prediction { get; set; }
        }

        public static void AnomalyDetectionWithSrCnnSingleFile(string filename, double threshold, double sensitivity)
        {
            var ml = new MLContext(1);
            IDataView dataView;
            string inputPath = @"D:\DataSet\nab_labeled\" + filename;
            string outputPath = @"D:\DataSet\output\nab_labeled_stl_sr_simple_pd\" + filename;

            List<string> inputs = new List<string>();
            //List<double> xs = new List<double>();
            List<TimeSeriesDataDouble> ys = new List<TimeSeriesDataDouble>();
            var header = "";

            using (var reader = new StreamReader(inputPath))
            {
                header = reader.ReadLine();

                //int count = 0;

                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    inputs.Add(line);
                    var values = line.Split(',');

                    ys.Add(new TimeSeriesDataDouble { Value = double.Parse(values[1]) });
                }
            }

            dataView = ml.Data.LoadFromEnumerable(ys);

            // Setup the detection arguments
            string outputColumnName = nameof(SrCnnAnomalyDetection.Prediction);
            string inputColumnName = nameof(TimeSeriesDataDouble.Value);

            Stopwatch watch = new Stopwatch();
            watch.Start();

            int batchSize = -1;
            string parameters = string.Format("threshold:{0};batchSize:{1};sensitivity:{2}", threshold, batchSize, sensitivity);

            // Do batch anomaly detection
            var outputDataView = ml.AnomalyDetection.DetectEntireAnomalyBySrCnn(dataView, outputColumnName, inputColumnName,
                threshold: threshold, batchSize: batchSize, sensitivity: sensitivity, SrCnnDetectMode.AnomalyAndMargin);

            // Getting the data of the newly created column as an IEnumerable of
            // SrCnnAnomalyDetection.
            var predictionColumn = ml.Data.CreateEnumerable<SrCnnAnomalyDetection>(
                outputDataView, reuseRowObject: false);

            using (var writer = new StreamWriter(outputPath))
            {
                writer.WriteLine(header + ",isAnomaly,expectedValue,upperBoundary,lowerBoundary,mag,unit," + parameters);

                int row = 0;
                foreach (var prediction in predictionColumn)
                {
                    writer.WriteLine("{0},{1},{2},{3},{4},{5},{6}", inputs[row], prediction.Prediction[0], prediction.Prediction[3],
                        prediction.Prediction[5], prediction.Prediction[6], prediction.Prediction[2], prediction.Prediction[4]);
                    ++row;
                }
            }
            watch.Stop();
            Console.WriteLine("Takes {0} milliseconds to finish.", watch.ElapsedMilliseconds);
        }

        public static void SrCnnPerfTest(int seriesLength)
        {
            string datasetPath = @"D:\DataSet\nab_yahoo_mix\";

            var ml = new MLContext(1);
            List<IDataView> dataViews = new List<IDataView>();

            foreach (var file in Directory.EnumerateFiles(datasetPath))
            {
                List<TimeSeriesDataDouble> ys = new List<TimeSeriesDataDouble>();
                var header = "";

                using var reader = new StreamReader(file);
                header = reader.ReadLine();

                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');

                    ys.Add(new TimeSeriesDataDouble { Value = double.Parse(values[1]) });

                    if (ys.Count == seriesLength)
                    {
                        dataViews.Add(ml.Data.LoadFromEnumerable(ys.ToArray()));
                        ys.Clear();
                    }
                }
                if (ys.Count == seriesLength)
                {
                    dataViews.Add(ml.Data.LoadFromEnumerable(ys.ToArray()));
                    ys.Clear();
                }
            }

            // Setup the detection arguments
            string outputColumnName = nameof(SrCnnAnomalyDetection.Prediction);
            string inputColumnName = nameof(TimeSeriesDataDouble.Value);

            double threshold = 0.3;
            int batchSize = -1;
            double sensitivity = 60;
            string parameters = string.Format("threshold:{0};batchSize:{1};sensitivity:{2}", threshold, batchSize, sensitivity);
            Stopwatch watch = new Stopwatch();
            watch.Start();

            foreach (var dataView in dataViews)
            {
                // Do batch anomaly detection
                var outputView = ml.AnomalyDetection.DetectEntireAnomalyBySrCnn(dataView, outputColumnName, inputColumnName,
                    threshold: threshold, batchSize: batchSize, sensitivity: sensitivity, SrCnnDetectMode.AnomalyAndExpectedValue);
                // Getting the data of the newly created column as an IEnumerable of
                // SrCnnAnomalyDetection.
                var predictionColumn = ml.Data.CreateEnumerable<SrCnnAnomalyDetection>(
                    outputView, reuseRowObject: false);

                foreach (var prediction in predictionColumn)
                {
                    _ = prediction.Prediction[0];
                    _ = prediction.Prediction[1];
                    _ = prediction.Prediction[2];
                }
            }

            watch.Stop();
            Console.WriteLine("Takes {0} milliseconds to process {1} lines with length {2}, average process time {3} ms."
                , watch.ElapsedMilliseconds, dataViews.Count, seriesLength, watch.ElapsedMilliseconds * 1.0 / (dataViews.Count));
        }

        public static void RunSR(string datasetPath, string outputRoot, double threshold, double sensitivity, int batchSize)
        {
            var subFolder = string.Format("b_{0}_s_{1}", batchSize, sensitivity);
            var fullFolderPath = Path.Combine(outputRoot, subFolder);

            if (!Directory.Exists(fullFolderPath))
            {
                Directory.CreateDirectory(fullFolderPath);
            }

            var ml = new MLContext(1);
            IDataView dataView;

            foreach (var file in Directory.EnumerateFiles(datasetPath))
            {
                List<string> inputs = new List<string>();
                List<TimeSeriesDataDouble> ys = new List<TimeSeriesDataDouble>();
                var header = "";

                using (var reader = new StreamReader(file))
                {
                    header = reader.ReadLine();

                    //int count = 0;

                    while (!reader.EndOfStream)
                    {
                        var line = reader.ReadLine();
                        inputs.Add(line);
                        var values = line.Split(',');

                        ys.Add(new TimeSeriesDataDouble { Value = double.Parse(values[1]) });
                    }
                }

                dataView = ml.Data.LoadFromEnumerable(ys);

                // Setup the detection arguments
                string outputColumnName = nameof(SrCnnAnomalyDetection.Prediction);
                string inputColumnName = nameof(TimeSeriesDataDouble.Value);

                // Do batch anomaly detection
                var outputDataView = ml.AnomalyDetection.DetectEntireAnomalyBySrCnn(dataView, outputColumnName, inputColumnName,
                    threshold: threshold, batchSize: batchSize, sensitivity: sensitivity, SrCnnDetectMode.AnomalyOnly);

                // Getting the data of the newly created column as an IEnumerable of
                // SrCnnAnomalyDetection.
                var predictionColumn = ml.Data.CreateEnumerable<SrCnnAnomalyDetection>(
                    outputDataView, reuseRowObject: false);

                var splitFilename = file.Split('\\');
                var outputPath = Path.Combine(fullFolderPath, splitFilename[splitFilename.Length - 1]);
                using StreamWriter writer = new StreamWriter(outputPath);

                writer.WriteLine(header + ",isAnomaly,expectedValue,upperBoundary,lowerBoundary");

                int row = 0;
                foreach (var prediction in predictionColumn)
                {
                    writer.WriteLine("{0},{1},{2},{3},{4}", inputs[row], prediction.Prediction[0], prediction.Prediction[3], prediction.Prediction[5], prediction.Prediction[6]);
                    ++row;
                }
            }
        }

        public static void QualityTest()
        {
            string datasetPath = @"D:\DataSet\yahoo\";
            string outputRoot = @"D:\DataSet\yahoo_sr_stl_result";

            List<double> threshold = new List<double>() { 0.30 };
            List<int> batchSize = new List<int>() { -1, 1024, 2048 };
            List<double> sensitivity = new List<double>() { 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0 };

            foreach (var t in threshold)
            {
                foreach (var b in batchSize)
                {
                    foreach (var s in sensitivity)
                    {
                        RunSR(datasetPath, outputRoot, t, s, b);
                    }
                }
            }
        }

        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            //PerfTest1();
            //AnomalyDetectionWithSrCnnSaveResults();
            //PerfTest2();
            foreach (var i in new int[] { 128, 256, 512, 1024, 2048, 4096, 8192, 10000 })
            {
                SrCnnPerfTest(i);
                //PerfTest4(i);
            }
            //QualityTest();
            //PerfTest4(4096);
            //SrCnnPerfTest(4096);

            //var fileNames = new string[]
            //{
            //            "art_daily_no_noise",
            //            "art_daily_perfect_square_wave",
            //            "art_daily_small_noise",
            //            "art_flatline",
            //            "art_noisy",
            //            "art_daily_flatmiddle",
            //            "art_daily_jumpsdown",
            //            "art_daily_jumpsup",
            //            "art_daily_nojump",
            //            "art_increase_spike_density",
            //            "art_load_balancer_spikes",
            //            "exchange-3_cpc_results",
            //            "exchange-3_cpm_results",
            //            "exchange-4_cpc_results",
            //            "exchange-4_cpm_results",
            //            "ec2_cpu_utilization_24ae8d",
            //            "ec2_cpu_utilization_53ea38",
            //            "ec2_cpu_utilization_5f5533",
            //            "ec2_cpu_utilization_77c1ca",
            //            "ec2_cpu_utilization_825cc2",
            //            "ec2_cpu_utilization_ac20cd",
            //            "ec2_cpu_utilization_c6585a",
            //            "ec2_cpu_utilization_fe7f93",
            //            "ec2_disk_write_bytes_c0d644",
            //            "ec2_network_in_257a54",
            //            "elb_request_count_8c0756",
            //            "grok_asg_anomaly",
            //            "iio_us-east-1_i-a2eb1cd9_NetworkIn",
            //            "rds_cpu_utilization_cc0c53",
            //            "rds_cpu_utilization_e47b3b",
            //            "ambient_temperature_system_failure",
            //            "rogue_agent_key_hold",
            //            "rogue_agent_key_updown",
            //            "occupancy_6005",
            //            "speed_6005",
            //            "speed_7578",
            //            "TravelTime_387",
            //            "TravelTime_451"
            //};
            //foreach(var file in fileNames)
            //{
            //    Console.WriteLine(file);
            //    AnomalyDetectionWithSrCnnSingleFile(file + ".csv", 0.3, 50);
            //}
            //int index = 36;
            //Console.WriteLine(fileNames[index]);
            //AnomalyDetectionWithSrCnnSingleFile(fileNames[index] + ".csv", 0.3, 50);
        }
    }
}
