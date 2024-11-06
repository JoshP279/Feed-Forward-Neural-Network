using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Assignment_2
{
    public class Program
    {
        static double[,] z;
        static int[] t;
        static HashSet<int> y;

        public static void Main()
        {
            LoadRawData("Plates.csv", out z, out t, out y);
            int inputSize = z.GetLength(1);

            double[,] z_val;
            int[] t_val;
            LoadValidationData(out z_val, out t_val, "TrainValidation.csv");
            FFNNModel model = new(
                z: z,
                t: t,
                input_size: inputSize,
                hidden_neurons: 35,
                output_size: y.Count(),
                learning_rate: 0.001,
                epochs: 20000,
                train: false,
                z_val: z_val,
                t_val: t_val
            );

            Console.WriteLine("\nComparing against training data...");
            ComparePredictions(model);

            LoadValidationData(out z_val, out t_val, "TestValidation.csv");
            Console.WriteLine("\nComparing against test data...");
            ComparePredictions(model, z_val, t_val);

            LoadEvalData("Evaluation.csv", out z);
            DisplayPredictions(model);
        }

        private static void ComparePredictions(FFNNModel model, double[,] z_data = null, int[] t_data = null)
        {
            z_data ??= z;
            t_data ??= t;

            int correct = 0;
            int wrong = 0;
            int[] o = new int[t_data.Length];

            Dictionary<int, int> correctPredictionsPerClass = new Dictionary<int, int>();
            Dictionary<int, int> totalClassificationsPerClass = new Dictionary<int, int>();

            for (int i = 1; i <= 7; i++)
            {
                correctPredictionsPerClass[i] = 0;
                totalClassificationsPerClass[i] = 0;
            }

            for (int i = 0; i < z_data.GetLength(0); i++)
            {
                double[] predictedOutput = model.Predict(GetCurrentPattern(z_data, i));
                int predictedClass = Array.IndexOf(predictedOutput, predictedOutput.Max()) + 1;
                o[i] = predictedClass;
                int actualClass = t_data[i];

                totalClassificationsPerClass[actualClass]++;

                if (predictedClass == actualClass)
                {
                    correct++;
                    correctPredictionsPerClass[actualClass]++;
                }
                else
                {
                    wrong++;
                }
            }

            double percentage = (double)correct / (correct + wrong) * 100;
            Console.WriteLine($"Correct: {correct} | Wrong: {wrong} | Accuracy: {percentage}%");
            Console.WriteLine($"SSE: {model.SSE(z_data, t_data)}");

            Console.WriteLine("Correct predictions and accuracy per class:");
            foreach (var kvp in correctPredictionsPerClass)
            {
                int classLabel = kvp.Key;
                int correctPredictions = kvp.Value;
                int totalClassifications = totalClassificationsPerClass[classLabel];
                double classAccuracy = (double)correctPredictions / totalClassifications * 100;
                Console.WriteLine($"Class {classLabel}: {correctPredictions} / {totalClassifications} ({classAccuracy}%)");
            }
        }

        private static void DisplayPredictions(FFNNModel model)
        {
            int j = 1;
            Console.WriteLine("\nPredicted values: ");
            for (int i = 0; i < z.GetLength(0); i++)
            {
                double[] predictedOutput = model.Predict(GetCurrentPattern(z, i));
                int predictedClass = Array.IndexOf(predictedOutput, predictedOutput.Max()) + 1;
                Console.WriteLine($"Predicted Pattern {j}: {predictedClass}");
                j++;
            }
        }

        public static double[] GetCurrentPattern(double[,] z, int p)
        {
            double[] z_p = new double[z.GetLength(1)];
            for (int i = 0; i < z.GetLength(1); i++)
            {
                z_p[i] = z[p, i];
            }
            return z_p;
        }
        public static void LoadValidationData(out double[,] z, out int[] t, String filePath)
        {
            string[] lines = File.ReadAllLines(filePath);
            int numLines = lines.Length;
            int numInputs = lines[0].Split(',').Length - 1;

            z = new double[numLines, numInputs];
            t = new int[numLines];
            y = new HashSet<int>();

            int[] classificationCounts = new int[7];


            for (int i = 0; i < numLines; i++)
            {
                string[] values = lines[i].Split(',');

                for (int j = 0; j < numInputs; j++)
                {
                    z[i, j] = double.Parse(values[j]);
                }

                t[i] = int.Parse(values[numInputs]);
                y.Add(t[i]);

                classificationCounts[t[i] - 1]++;
            }

            NormaliseData(z);
        }

        public static void LoadRawData(string filePath, out double[,] z, out int[] t, out HashSet<int> y, double noiseLevel = 0.1)
        {
            string[] lines = File.ReadAllLines(filePath);
            int numLines = lines.Length;
            int numInputs = lines[0].Split(',').Length - 1;

            z = new double[numLines, numInputs];
            t = new int[numLines];
            y = new HashSet<int>();
            Random rand = new Random();

            for (int i = 0; i < numLines; i++)
            {
                string[] values = lines[i].Split(',');

                for (int j = 0; j < numInputs; j++)
                {
                    z[i, j] = double.Parse(values[j]);
                    z[i, j] += (rand.NextDouble() - 0.5) * noiseLevel;
                }

                t[i] = int.Parse(values[numInputs]);
                y.Add(t[i]);
            }
            NormaliseData(z);
        }

        private static void LoadEvalData(string filePath, out double[,] z)
        {
            string[] lines = File.ReadAllLines(filePath);
            int numLines = lines.Length;
            int numInputs = lines[0].Split(',').Length;

            z = new double[numLines, numInputs];
            for (int i = 0; i < numLines; i++)
            {
                string[] values = lines[i].Split(',');

                for (int j = 0; j < numInputs; j++)
                {
                    z[i, j] = double.Parse(values[j]);
                }
            }
            NormaliseData(z);
        }
        private static void NormaliseData(double[,] z)
        {
            int numLines = z.GetLength(0);
            int numInputs = z.GetLength(1);

            for (int j = 0; j < numInputs; j++)
            {
                double min = double.MaxValue;
                double max = double.MinValue;

                for (int i = 0; i < numLines; i++)
                {
                    if (z[i, j] < min)
                        min = z[i, j];
                    if (z[i, j] > max)
                        max = z[i, j];
                }

                for (int i = 0; i < numLines; i++)
                {
                    z[i, j] = (z[i, j] - min) / (max - min);
                }
            }
        }
    }
}