namespace Assignment_2
{

    public class FFNNModel
    {
        private double[,] w; // Weights for output layer neurons
        private double[,] v; // Weights for hidden layer neurons
        private int J; // Number of hidden layer neurons
        private int K; // Number of output layer neurons
        private int I; // Number of input units
        private double learning_rate;

        public FFNNModel(double[,] z, int[] t, int input_size, int hidden_neurons, int output_size, double learning_rate, int epochs, bool train, double[,] z_val = null, int[] t_val = null)
        {
            this.I = input_size;
            this.J = hidden_neurons;
            this.K = output_size;
            this.learning_rate = learning_rate;
            w = InitWeights(K, J + 1, -1 / Math.Sqrt(I), 1 / Math.Sqrt(I));
            v = InitWeights(J, I + 1, -1 / Math.Sqrt(I), 1 / Math.Sqrt(I));
            if (!train)
            {
                LoadWeights("outputWeights.csv", w);
                LoadWeights("hiddenWeights.csv", v);
            }
            else GradientDescent(z, t, epochs, z_val, t_val);
        }

        private double[,] InitWeights(int rows, int cols, double lb, double ub)
        {
            Random rand = new Random();
            double[,] weights = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    weights[i, j] = lb + rand.NextDouble() * (ub - lb);
                    //weights[i, j] = rand.NextDouble() * 0.1;
                }
            }
            return weights;
        }

        public double SSE(double[,] z, int[] t)
        {
            double sum = 0;
            for (int p = 0; p < z.GetLength(0); p++)
            {
                double[] z_p = GetCurrentPattern(z, p);
                double[] y = CalculateHiddenLayerOutputs(z_p);
                double[] o = CalculateOutputLayerOutputs(y);

                double[] target = new double[o.Length];
                target[t[p] - 1] = 1.0;

                for (int k = 0; k < K; k++)
                {
                    sum += Math.Pow(target[k] - o[k], 2);
                }
            }
            return 0.5 * sum;
        }

        private double[] CalculateHiddenLayerOutputs(double[] z_p)
        {
            double[] y = new double[this.J];
            for (int j = 0; j < J; j++)
            {
                double net_yj = 0;
                for (int i = 0; i < I; i++)
                {
                    net_yj += v[j, i] * z_p[i];
                }
                net_yj += v[j, I];
                y[j] = Sigmoid(net_yj);
            }
            return y;
        }

        private double[] CalculateOutputLayerOutputs(double[] y)
        {
            double[] o = new double[this.K];
            for (int k = 0; k < K; k++)
            {
                double net_ok = 0;
                for (int j = 0; j < J; j++)
                {
                    net_ok += w[k, j] * y[j];
                }
                net_ok += w[k, J];
                o[k] = Sigmoid(net_ok);
            }
            return o;
        }

        public void GradientDescent(double[,] z, int[] t, int max_epoch, double[,] z_val, int[] t_val)
        {
            Console.WriteLine("Training model...");
            int cur_epoch = 0;
            double[] sse_history = new double[max_epoch];
            List<double> val_errors = new List<double>();

            while (cur_epoch < max_epoch)
            {
                int misclassifications = 0;
                double total_error = 0;

                for (int p = 0; p < z.GetLength(0); p++)
                {
                    double[] z_p = GetCurrentPattern(z, p);
                    double[] y = CalculateHiddenLayerOutputs(z_p);
                    double[] o = CalculateOutputLayerOutputs(y);

                    int predictedClass = Array.IndexOf(o, o.Max()) + 1;

                    if (predictedClass != t[p]) misclassifications++;

                    double[] target = new double[o.Length];
                    target[t[p] - 1] = 1.0;
                    for (int k = 0; k < K; k++)
                    {
                        double error = target[k] - o[k];
                        total_error += Math.Pow(error, 2);

                        for (int j = 0; j < J; j++)
                        {
                            w[k, j] -= learning_rate * -2 * error * o[k] * (1 - o[k]) * y[j];
                        }
                        w[k, J] -= learning_rate * -2 * error * o[k] * (1 - o[k]);
                    }

                    for (int j = 0; j < J; j++)
                    {
                        double delta_v_bias = 0;
                        for (int i = 0; i < I; i++)
                        {
                            double delta_v = 0;
                            for (int k = 0; k < K; k++)
                            {
                                double error = target[k] - o[k];
                                delta_v += -2 * error * o[k] * (1 - o[k]) * w[k, j] * y[j] * (1 - y[j]) * z_p[i];
                                delta_v_bias += -2 * error * o[k] * (1 - o[k]) * w[k, j] * y[j] * (1 - y[j]);
                            }
                            v[j, i] -= learning_rate * delta_v;
                        }
                        v[j, I] -= learning_rate * delta_v_bias;
                    }
                }

                double current_error = (0.5 * total_error);
                double val_error = Validate(z_val, t_val);
                val_errors.Add(val_error);
                double val_mean = val_errors.Average();
                double std_val_error = Math.Sqrt(val_errors.Average(v => Math.Pow(v - val_mean, 2) / val_errors.Count));

                if (val_error > (val_mean + std_val_error))
                {
                    Console.WriteLine("Overfitting on validation SSE, breaking early...");
                    break;
                }

                if (cur_epoch % 10 == 0)
                {
                    Console.WriteLine($"Epoch: {cur_epoch} | Learning Rate: {learning_rate} | Training SSE: {current_error} | Validation SSE: {val_error} | Stopping value {val_mean + std_val_error}");
                }
                sse_history[cur_epoch] = 0.5 * total_error;
                cur_epoch++;
            }
            SaveWeights("outputWeights.csv", w);
            SaveWeights("hiddenWeights.csv", v);
            SaveSSE("sseOverEpochs.csv", sse_history);
        }

        private double Validate(double[,] z_val, int[] t_val)
        {
            double total_error = 0;
            for (int p = 0; p < z_val.GetLength(0); p++)
            {
                double[] z_p = GetCurrentPattern(z_val, p);
                double[] y = CalculateHiddenLayerOutputs(z_p);
                double[] o = CalculateOutputLayerOutputs(y);

                double[] target = new double[o.Length];
                target[t_val[p] - 1] = 1.0;
                for (int k = 0; k < K; k++)
                {
                    double error = target[k] - o[k];
                    total_error += Math.Pow(error, 2);
                }
            }
            return (0.5 * total_error);
        }

        private void SaveSSE(string fileName, double[] sse_history)
        {
            using (StreamWriter streamWriter = new StreamWriter(fileName))
            {
                for (int i = 0; i < sse_history.Length; i++)
                {
                    streamWriter.Write(sse_history[i]);
                    if (i != sse_history.Length - 1)
                    {
                        streamWriter.Write(",");
                    }
                }
            }
            Console.WriteLine($"Saved {fileName} successfully!");
        }

        public double[] Predict(double[] z_p)
        {
            double[] y = CalculateHiddenLayerOutputs(z_p);
            return CalculateOutputLayerOutputs(y);
        }

        private double Sigmoid(double x) { return 1.0 / (1.0 + Math.Exp(-x)); }


        private double[] GetCurrentPattern(double[,] z, int p)
        {
            double[] z_p = new double[z.GetLength(1)];
            for (int i = 0; i < z.GetLength(1); i++)
            {
                z_p[i] = z[p, i];
            }
            return z_p;
        }

        private void SaveWeights(string fileName, double[,] weights)
        {
            using (StreamWriter streamWriter = new StreamWriter(fileName))
            {
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    for (int j = 0; j < weights.GetLength(1); j++)
                    {
                        streamWriter.Write(weights[i, j]);
                        if (j != weights.GetLength(1) - 1)
                        {
                            streamWriter.Write(",");
                        }
                    }
                    streamWriter.WriteLine();
                }
            }
            Console.WriteLine($"Saved {fileName} successfully!");
        }

        private void LoadWeights(string fileName, double[,] weights)
        {
            Console.WriteLine($"Loading {fileName}...");
            using (StreamReader streamReader = new StreamReader(fileName))
            {
                int rows = weights.GetLength(0);
                int cols = weights.GetLength(1);
                string line;
                int i = 0;

                while ((line = streamReader.ReadLine()) != null && i < rows)
                {
                    string[] values = line.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    for (int j = 0; j < cols; j++)
                    {
                        weights[i, j] = double.Parse(values[j]);
                    }
                    i++;
                }
            }
        }
    }
}