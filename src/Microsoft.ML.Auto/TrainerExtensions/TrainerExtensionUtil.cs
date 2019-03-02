﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;

namespace Microsoft.ML.Auto
{
    internal enum TrainerName
    {
        AveragedPerceptronBinary,
        AveragedPerceptronOva,
        FastForestBinary,
        FastForestOva,
        FastForestRegression,
        FastTreeBinary,
        FastTreeOva,
        FastTreeRegression,
        FastTreeTweedieRegression,
        LightGbmBinary,
        LightGbmMulti,
        LightGbmRegression,
        LinearSvmBinary,
        LinearSvmOva,
        LogisticRegressionBinary,
        LogisticRegressionOva,
        LogisticRegressionMulti,
        OnlineGradientDescentRegression,
        OrdinaryLeastSquaresRegression,
        PoissonRegression,
        SdcaBinary,
        SdcaMulti,
        SdcaRegression,
        StochasticGradientDescentBinary,
        StochasticGradientDescentOva,
        SymSgdBinary,
        SymSgdOva
    }

    internal static class TrainerExtensionUtil
    {
        private const string WeightColumn = "WeightColumn";
        private const string LabelColumn = "LabelColumn";

        public static T CreateOptions<T>(IEnumerable<SweepableParam> sweepParams, string labelColumn) where T : LearnerInputBaseWithLabel
        {
            var options = Activator.CreateInstance<T>();
            options.LabelColumn = labelColumn;
            if (sweepParams != null)
            {
                UpdateFields(options, sweepParams);
            }
            return options;
        }

        private static string[] _lightGbmTreeBoosterParamNames = new[] { "RegLambda", "RegAlpha" };
        private const string LightGbmTreeBoosterPropName = "Booster";

        public static LightGBM.Options CreateLightGbmOptions(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            var options = new LightGBM.Options();
            options.LabelColumn = columnInfo.LabelColumn;
            options.WeightColumn = columnInfo.WeightColumn;
            if (sweepParams != null)
            {
                var treeBoosterParams = sweepParams.Where(p => _lightGbmTreeBoosterParamNames.Contains(p.Name));
                var parentArgParams = sweepParams.Except(treeBoosterParams);
                UpdateFields(options, parentArgParams);
                UpdateFields(options.Booster, treeBoosterParams);
            }
            return options;
        }

        public static PipelineNode BuildOvaPipelineNode(ITrainerExtension multiExtension, ITrainerExtension binaryExtension,
            IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            var ovaNode = binaryExtension.CreatePipelineNode(sweepParams, columnInfo);
            ovaNode.Name = TrainerExtensionCatalog.GetTrainerName(multiExtension).ToString();
            return ovaNode;
        }

        public static PipelineNode BuildPipelineNode(TrainerName trainerName, IEnumerable<SweepableParam> sweepParams,
            string labelColumn, string weightColumn = null, IDictionary<string, object> additionalProperties = null)
        {
            var properties = BuildBasePipelineNodeProps(sweepParams, labelColumn, weightColumn);

            if (additionalProperties != null)
            {
                foreach (var property in additionalProperties)
                {
                    properties[property.Key] = property.Value;
                }
            }

            return new PipelineNode(trainerName.ToString(), PipelineNodeType.Trainer, DefaultColumnNames.Features,
                DefaultColumnNames.Score, properties);
        }

        public static PipelineNode BuildLightGbmPipelineNode(TrainerName trainerName, IEnumerable<SweepableParam> sweepParams,
            string labelColumn, string weightColumn)
        {
            return new PipelineNode(trainerName.ToString(), PipelineNodeType.Trainer, DefaultColumnNames.Features,
                DefaultColumnNames.Score, BuildLightGbmPipelineNodeProps(sweepParams, labelColumn, weightColumn));
        }

        private static IDictionary<string, object> BuildBasePipelineNodeProps(IEnumerable<SweepableParam> sweepParams,
            string labelColumn, string weightColumn)
        {
            var props = new Dictionary<string, object>();
            if (sweepParams != null)
            {
                foreach (var sweepParam in sweepParams)
                {
                    props[sweepParam.Name] = sweepParam.ProcessedValue();
                }
            }
            props[LabelColumn] = labelColumn;
            if (weightColumn != null)
            {
                props[WeightColumn] = weightColumn;
            }
            return props;
        }

        private static IDictionary<string, object> BuildLightGbmPipelineNodeProps(IEnumerable<SweepableParam> sweepParams,
            string labelColumn, string weightColumn)
        {
            Dictionary<string, object> props = null;
            if (sweepParams == null)
            {
                props = new Dictionary<string, object>();
            }
            else
            {
                var treeBoosterParams = sweepParams.Where(p => _lightGbmTreeBoosterParamNames.Contains(p.Name));
                var parentArgParams = sweepParams.Except(treeBoosterParams);

                var treeBoosterProps = treeBoosterParams.ToDictionary(p => p.Name, p => (object)p.ProcessedValue());
                var treeBoosterCustomProp = new CustomProperty("Options.TreeBooster.Options", treeBoosterProps);

                props = parentArgParams.ToDictionary(p => p.Name, p => (object)p.ProcessedValue());
                props[LightGbmTreeBoosterPropName] = treeBoosterCustomProp;
            }

            props[LabelColumn] = labelColumn;
            if (weightColumn != null)
            {
                props[WeightColumn] = weightColumn;
            }

            return props;
        }

        public static ParameterSet BuildParameterSet(TrainerName trainerName, IDictionary<string, object> props)
        {
            props = props.Where(p => p.Key != LabelColumn && p.Key != WeightColumn)
                .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

            if (trainerName == TrainerName.LightGbmBinary || trainerName == TrainerName.LightGbmMulti ||
                trainerName == TrainerName.LightGbmRegression)
            {
                return BuildLightGbmParameterSet(props);
            }

            var paramVals = props.Select(p => new StringParameterValue(p.Key, p.Value.ToString()));
            return new ParameterSet(paramVals);
        }

        public static ColumnInformation BuildColumnInfo(IDictionary<string, object> props)
        {
            var columnInfo = new ColumnInformation();

            columnInfo.LabelColumn = props[LabelColumn] as string;

            props.TryGetValue(WeightColumn, out var weightColumn);
            columnInfo.WeightColumn = weightColumn as string;

            return columnInfo;
        }

        private static ParameterSet BuildLightGbmParameterSet(IDictionary<string, object> props)
        {
            var parentProps = props.Where(p => p.Key != LightGbmTreeBoosterPropName);
            var treeProps = ((CustomProperty)props[LightGbmTreeBoosterPropName]).Properties;
            var allProps = parentProps.Union(treeProps);
            var paramVals = allProps.Select(p => new StringParameterValue(p.Key, p.Value.ToString()));
            return new ParameterSet(paramVals);
        }

        private static void SetValue(FieldInfo fi, IComparable value, object obj, Type propertyType)
        {
            if (propertyType == value?.GetType())
                fi.SetValue(obj, value);
            else if (propertyType == typeof(double) && value is float)
                fi.SetValue(obj, Convert.ToDouble(value));
            else if (propertyType == typeof(int) && value is long)
                fi.SetValue(obj, Convert.ToInt32(value));
            else if (propertyType == typeof(long) && value is int)
                fi.SetValue(obj, Convert.ToInt64(value));
        }

        /// <summary>
        /// Updates properties of object instance based on the values in sweepParams
        /// </summary>
        public static void UpdateFields(object obj, IEnumerable<SweepableParam> sweepParams)
        {
            foreach (var param in sweepParams)
            {
                try
                {
                    // Only updates property if param.value isn't null and
                    // param has a name of property.
                    if (param.RawValue == null)
                    {
                        continue;
                    }
                    var fi = obj.GetType().GetField(param.Name);
                    var propType = Nullable.GetUnderlyingType(fi.FieldType) ?? fi.FieldType;

                    if (param is SweepableDiscreteParam dp)
                    {
                        var optIndex = (int)dp.RawValue;
                        //Contracts.Assert(0 <= optIndex && optIndex < dp.Options.Length, $"Options index out of range: {optIndex}");
                        var option = dp.Options[optIndex].ToString().ToLower();

                        // Handle <Auto> string values in sweep params
                        if (option == "auto" || option == "<auto>" || option == "< auto >")
                        {
                            //Check if nullable type, in which case 'null' is the auto value.
                            if (Nullable.GetUnderlyingType(fi.FieldType) != null)
                                fi.SetValue(obj, null);
                            else if (fi.FieldType.IsEnum)
                            {
                                // Check if there is an enum option named Auto
                                var enumDict = fi.FieldType.GetEnumValues().Cast<int>()
                                    .ToDictionary(v => Enum.GetName(fi.FieldType, v), v => v);
                                if (enumDict.ContainsKey("Auto"))
                                    fi.SetValue(obj, enumDict["Auto"]);
                            }
                        }
                        else
                            SetValue(fi, (IComparable)dp.Options[optIndex], obj, propType);
                    }
                    else
                        SetValue(fi, param.RawValue, obj, propType);
                }
                catch (Exception)
                {
                    throw new InvalidOperationException($"Cannot set parameter {param.Name} for {obj.GetType()}");
                }
            }
        }

        public static TrainerName GetTrainerName(BinaryClassificationTrainer binaryTrainer)
        {
            switch (binaryTrainer)
            {
                case BinaryClassificationTrainer.AveragedPerceptron:
                    return TrainerName.AveragedPerceptronBinary;
                case BinaryClassificationTrainer.FastForest:
                    return TrainerName.FastForestBinary;
                case BinaryClassificationTrainer.FastTree:
                    return TrainerName.FastTreeBinary;
                case BinaryClassificationTrainer.LightGbm:
                    return TrainerName.LightGbmBinary;
                case BinaryClassificationTrainer.LinearSupportVectorMachines:
                    return TrainerName.LinearSvmBinary;
                case BinaryClassificationTrainer.LogisticRegression:
                    return TrainerName.LogisticRegressionBinary;
                case BinaryClassificationTrainer.StochasticDualCoordinateAscent:
                    return TrainerName.SdcaBinary;
                case BinaryClassificationTrainer.StochasticGradientDescent:
                    return TrainerName.StochasticGradientDescentBinary;
                case BinaryClassificationTrainer.SymbolicStochasticGradientDescent:
                    return TrainerName.SymSgdBinary;
            }

            // never expected to reach here
            throw new NotSupportedException($"{binaryTrainer} not supported");
        }

        public static TrainerName GetTrainerName(MulticlassClassificationTrainer multiTrainer)
        {
            switch (multiTrainer)
            {
                case MulticlassClassificationTrainer.AveragedPerceptronOVA:
                    return TrainerName.AveragedPerceptronOva;
                case MulticlassClassificationTrainer.FastForestOVA:
                    return TrainerName.FastForestOva;
                case MulticlassClassificationTrainer.FastTreeOVA:
                    return TrainerName.FastTreeOva;
                case MulticlassClassificationTrainer.LightGbm:
                    return TrainerName.LightGbmMulti;
                case MulticlassClassificationTrainer.LinearSupportVectorMachinesOVA:
                    return TrainerName.LinearSvmOva;
                case MulticlassClassificationTrainer.LogisticRegression:
                    return TrainerName.LogisticRegressionMulti;
                case MulticlassClassificationTrainer.LogisticRegressionOVA:
                    return TrainerName.LogisticRegressionOva;
                case MulticlassClassificationTrainer.StochasticDualCoordinateAscent:
                    return TrainerName.SdcaMulti;
                case MulticlassClassificationTrainer.StochasticGradientDescentOVA:
                    return TrainerName.StochasticGradientDescentOva;
                case MulticlassClassificationTrainer.SymbolicStochasticGradientDescentOVA:
                    return TrainerName.SymSgdOva;
            }

            // never expected to reach here
            throw new NotSupportedException($"{multiTrainer} not supported");
        }

        public static TrainerName GetTrainerName(RegressionTrainer regressionTrainer)
        {
            switch (regressionTrainer)
            {
                case RegressionTrainer.FastForest:
                    return TrainerName.FastForestRegression;
                case RegressionTrainer.FastTree:
                    return TrainerName.FastTreeRegression;
                case RegressionTrainer.FastTreeTweedie:
                    return TrainerName.FastTreeTweedieRegression;
                case RegressionTrainer.LightGbm:
                    return TrainerName.LightGbmRegression;
                case RegressionTrainer.OnlineGradientDescent:
                    return TrainerName.OnlineGradientDescentRegression;
                case RegressionTrainer.OrdinaryLeastSquares:
                    return TrainerName.OrdinaryLeastSquaresRegression;
                case RegressionTrainer.PoissonRegression:
                    return TrainerName.PoissonRegression;
                case RegressionTrainer.StochasticDualCoordinateAscent:
                    return TrainerName.SdcaRegression;
            }

            // never expected to reach here
            throw new NotSupportedException($"{regressionTrainer} not supported");
        }

        public static IEnumerable<TrainerName> GetTrainerNames(IEnumerable<BinaryClassificationTrainer> binaryTrainers)
        {
            return binaryTrainers?.Select(t => GetTrainerName(t));
        }

        public static IEnumerable<TrainerName> GetTrainerNames(IEnumerable<MulticlassClassificationTrainer> multiTrainers)
        {
            return multiTrainers?.Select(t => GetTrainerName(t));
        }

        public static IEnumerable<TrainerName> GetTrainerNames(IEnumerable<RegressionTrainer> regressionTrainers)
        {
            return regressionTrainers?.Select(t => GetTrainerName(t));
        }
    }
}
