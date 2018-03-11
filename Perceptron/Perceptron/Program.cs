using System;
namespace Perceptron
{
    class Program
    {
        static void Main(string[] args)
        {
            //Simulando uma soma
            double[][] entradas =
            {
                new double[] {0.2, 0.1},
                new double[] {0.5, 0.2},
                new double[] {0.1, 0.8}
            };

            double[][] saidas =
            {
                new double[] {0.3},
                new double[] {0.7},
                new double[] {0.9}
            };

            Percepton net = new Percepton(2,1);

            for (int i = 0; i < 1000; i++)
            {
                for (int k = 0; k < entradas.Length; k++)
                {
                    net.Treinar(entradas[k], saidas[k]);

                    Console.WriteLine("Saída - {0} -> Alvo - {1}", net.Computar(entradas[k])[0], saidas[k][0]);
                }
            }
            Console.ReadLine();

            #region TesteRedeTreinada
            double[][] novasEntradas =
            {
                new double[] {0.2, 0.1},
                new double[] {0.1, 0.1},
                new double[] {0.3, 0.5},
                new double[] {0.4, 0.5},
                new double[] {0.2, 0.6},
                new double[] {0.5, 0.5}
            };

            for (int k = 0; k < novasEntradas.Length; k++)
            {
                Console.WriteLine("{0} + {1} = {2} ", novasEntradas[k][0], novasEntradas[k][1], net.Computar(novasEntradas[k])[0]);
            }

            Console.ReadLine();
            #endregion
        }
    }

    /// <summary>
    /// Class que implementa uma Rede Neural do tipo Percepton
    /// </summary>
    public class Percepton
    {
        // Camada de entrada x[] camada de saída y[]
        // w[] e b[]
        private double[] bias;
        private double[,] pesos;

        private int Entrada;
        private int Saida;

        public Percepton(int entradas, int saidas)
        {
            bias = new double[saidas];

            this.Entrada = entradas;
            this.Saida = saidas;

            pesos = new double[saidas, entradas];
            Random rnd = new Random(); 
            for (int i = 0; i < entradas; i++)
            {
                for (int j = 0; j < saidas; j++)
                {
                    pesos[j, i] = rnd.NextDouble();
                }
            }
        }

        public void Treinar(double[] entrada, double[] alvo)
        {
            double[] atual = Computar(entrada);
            for (int i = 0; i < Saida; i++)
            {
                for (int j = 0; j < Entrada; j++)
                {
                    pesos[i, j] += (alvo[i] - atual[i]) * entrada[j];
                }
                bias[i] += (alvo[i] - atual[i]);

            }            
        }

        public double[] Computar(params double[] entrada)
        {
            double[] saida = new double[Saida];

            for (int i = 0; i < Saida; i++)
            {
                double soma = 0.0;
                for (int j = 0; j < entrada.Length; j++)
                {
                    soma += pesos[i, j] * entrada[j];
                }
                soma += bias[i];
                saida[i] = Segmoid(soma);
            }
            return saida;
        }

        private double Segmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
    }
}
