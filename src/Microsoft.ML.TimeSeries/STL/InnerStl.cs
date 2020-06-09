﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TimeSeries
{
    internal class StlConfiguration
    {
        /// <summary>
        /// the smoothing parameter for the seasonal component.
        /// should be odd, and at least 7.
        /// </summary>
        public const int Ns = 9;

        /// <summary>
        /// the number of passes through the inner loop. /ref this value is set to 2, which works for many cases
        /// </summary>
        public const int Ni = 2;

        /// <summary>
        /// the number of robustness iterations of the outer loop
        /// </summary>
        public const int No = 10;

        public StlConfiguration()
        {
            Np = -1;
        }

        public StlConfiguration(int np)
        {
            Np = np;
        }

        /// <summary>
        /// the number of observations in each cycle of the seasonal component
        /// </summary>
        public int Np { get; }

        /// <summary>
        /// the smoothing parameter for the low-pass filter.
        /// /ref: should be the least odd integer greater than or equal to np.
        /// it will preventing the trend and seasonal components from competing for the same variation in the data.
        /// </summary>
        public int Nl
        {
            get
            {
                if (Np % 2 == 0)
                    return Np + 1;
                return Np;
            }
        }

        /// <summary>
        /// the smoothing parameter for the trend component.
        /// /ref: in order to avoid the trend ans seasonal components compete for variation in the data, the nt should be chosen
        /// s.t., satisty the following inequality.
        /// </summary>
        public int Nt
        {
            get
            {
                double value = 1.5 * Np / (1.0 - 1.5 / StlConfiguration.Ns);
                int result = (int)value + 1;
                if (result % 2 == 0)
                    result++;
                return result;
            }
        }
    }

    internal class InnerStl
    {
        private readonly IReadOnlyList<double> _x;
        private readonly IReadOnlyList<double> _y;
        private readonly int _length;
        private readonly bool _isTemporal;
        private readonly StlConfiguration _config;

        private readonly double[] _seasonalComponent;
        private readonly double[] _trendComponent;
        private readonly double[] _residual;
        private readonly int[] _outlierIndexes;
        private readonly double[] _outlierSeverity;

        /// <summary>
        /// Initializes a new instance of the <see cref="InnerStl"/> class.
        /// for a time series, only with y values. assume the x-values are 0, 1, 2, ...
        /// since this method supports decompose seasonal signal, which requires the equal-space of the input x-axis values.
        /// otherwise, the smoothing on seasonal component will be very complicated.
        /// </summary>
        /// <param name="yValues">the y-axis values</param>
        /// <param name="config">the configuration for applying regression</param>
        /// <param name="isTemporal">if the regression is considered to take temporal information into account. in general, this is true if we are regressing a time series, and false if we are regressing scatter plot data</param>
        public InnerStl(IReadOnlyList<double> yValues, StlConfiguration config, bool isTemporal)
        {
            Contracts.CheckValue(yValues, nameof(yValues));
            Contracts.CheckValue(config, nameof(config));

            if (yValues.Count == 0)
                throw Contracts.Except("input data structure cannot be 0-length: innerSTL");

            _y = yValues;
            _length = _y.Count;
            _isTemporal = isTemporal;
            _x = VirtualXValuesProvider.GetXValues(_length);
            _config = config;

            _seasonalComponent = new double[_length];
            _trendComponent = new double[_length];
            _residual = new double[_length];
            _outlierIndexes = new int[_length];
            _outlierSeverity = new double[_length];
        }

        /// <summary>
        /// the seasonal component
        /// </summary>
        public IReadOnlyList<double> SeasonalComponent
        {
            get { return _seasonalComponent; }
        }

        /// <summary>
        /// the trend component
        /// </summary>
        public IReadOnlyList<double> TrendComponent
        {
            get { return _trendComponent; }
        }

        /// <summary>
        /// the left component after seasonal and trend are eliminated.
        /// </summary>
        public IReadOnlyList<double> Residual
        {
            get { return _residual; }
        }

        /// <summary>
        /// calculate the slope of the trend component
        /// </summary>
        public double Slope
        {
            get;
            private set;
        }

        /// <summary>
        /// the core for the robust trend-seasonal decomposition. see the ref: http://www.wessa.net/download/stl.pdf,
        /// see section 2 and 3. especially section 2.
        /// </summary>
        /// <returns>return true if the process goes successfully. otherwise, return false.</returns>
        public bool Decomposition()
        {
            double[] s = new double[_length];
            double[] t = new double[_length];
            for (int iter = 0; iter < StlConfiguration.Ni; iter++)
            {
                // step1: detrending
                double[] detrendedY = new double[_length];
                for (int i = 0; i < _length; i++)
                    detrendedY[i] = _y[i] - t[i];

                // step2: cycle-subseries smoothing
                List<double>[] cycleSubSeries = new List<double>[_config.Np];
                List<double>[] smoothedSubseries = new List<double>[_config.Np];
                for (int i = 0; i < _config.Np; i++)
                {
                    cycleSubSeries[i] = new List<double>();
                    smoothedSubseries[i] = new List<double>();
                }

                // obtain all the subseries
                for (int i = 0; i < _length; i++)
                {
                    int cycleIndex = i % _config.Np;
                    cycleSubSeries[cycleIndex].Add(detrendedY[i]);
                }

                // smoothing on each subseries
                for (int i = 0; i < cycleSubSeries.Length; i++)
                {
                    List<double> virtualXValues = VirtualXValuesProvider.GetXValues(cycleSubSeries[i].Count);

                    FastLoess model = new FastLoess(virtualXValues, cycleSubSeries[i], _isTemporal, StlConfiguration.Ns);
                    model.Estimate();

                    // add a prior point
                    smoothedSubseries[i].Add(model.EstimateY(-1.0));
                    smoothedSubseries[i].AddRange(model.Y);

                    // add a after point
                    smoothedSubseries[i].Add(model.EstimateY(cycleSubSeries[i].Count * 1.0));
                }

                // c is the smoothed series, with _length+2Np points.
                List<double> c = new List<double>();
                for (int i = 0; i < smoothedSubseries[0].Count; i++)
                {
                    for (int j = 0; j < smoothedSubseries.Length; j++)
                    {
                        if (smoothedSubseries[j].Count <= i)
                            break;
                        if (smoothedSubseries[j][i].Equals(double.NaN))
                        {
                            return false;
                        }
                        c.Add(smoothedSubseries[j][i]);
                    }
                }

                // step3: low-pass filtering of smoothed cycle-subseries
                List<double> c1 = MovingAverage(c, _config.Np);
                List<double> c2 = MovingAverage(c1, _config.Np);
                List<double> c3 = MovingAverage(c2, 3);
                List<double> virtualC3XValues = VirtualXValuesProvider.GetXValues(c3.Count);
                FastLoess lowPass = new FastLoess(virtualC3XValues, c3, _isTemporal, _config.Nl);
                lowPass.Estimate();

                // step4: detrending of smoothed cycle-subseries
                for (int i = 0; i < _length; i++)
                {
                    s[i] = c[i] - lowPass.Y[i];
                }

                // step5: deseasonalizing
                List<double> deseasonSeries = new List<double>();
                for (int i = 0; i < _length; i++)
                {
                    deseasonSeries.Add(_y[i] - s[i]);
                }

                // step6: trend smoothing
                List<double> virtualDeseasonSeries = VirtualXValuesProvider.GetXValues(deseasonSeries.Count);
                FastLoess trender = new FastLoess(virtualDeseasonSeries, deseasonSeries, _isTemporal, _config.Nt);
                trender.Estimate();
                for (int i = 0; i < _length; i++)
                {
                    t[i] = trender.Y[i];
                }
            }

            for (int i = 0; i < s.Length; i++)
            {
                _seasonalComponent[i] = s[i];
                _trendComponent[i] = t[i];
            }

            // the slope is still based on the regression models.
            Slope = (_trendComponent[_length - 1] - _seasonalComponent[0]) / (_length - 1);

            var absResiduals = new List<double>(_residual);
            for (int i = 0; i < _y.Count; i++)
            {
                _residual[i] = _y[i] - _seasonalComponent[i] - _trendComponent[i];
                absResiduals.Add(Math.Abs(_y[i] - _seasonalComponent[i] - _trendComponent[i]));
            }

            return true;
        }

        /// <summary>
        /// this class provides the virtual x values for multi object usage.
        /// the cache mechanism is used for performance consideration.
        /// </summary>
        internal class VirtualXValuesProvider
        {
            private static Dictionary<int, List<double>> _xValuesPool;

            static VirtualXValuesProvider()
            {
                _xValuesPool = new Dictionary<int, List<double>>();
            }

            /// <summary>
            /// get a list of virtual x-axis values. the values are from 0 to length - 1.
            /// </summary>
            /// <param name="length">specify the length you want to create the x values.</param>
            /// <returns>if this is cached, return directly. otherwise, create a new list and return</returns>
            internal static List<double> GetXValues(int length)
            {
                lock (_xValuesPool)
                {
                    List<double> xValues;
                    if (_xValuesPool.TryGetValue(length, out xValues))
                        return xValues;

                    var newXValues = new List<double>(length);
                    for (int i = 0; i < length; i++)
                        newXValues.Add(i);

                    _xValuesPool.Add(length, newXValues);
                    return newXValues;
                }
            }
        }

        private static List<double> MovingAverage(IReadOnlyList<double> s, int length)
        {
            List<double> results = new List<double>(s.Count);
            double partialSum = 0;
            for (int i = 0; i < length; ++i)
            {
                partialSum += s[i];
            }

            for (int i = length; i < s.Count; ++i)
            {
                results.Add(partialSum / length);
                partialSum = partialSum - s[i - length] + s[i];
            }
            results.Add(partialSum / length);

            return results;
        }
    }
}
