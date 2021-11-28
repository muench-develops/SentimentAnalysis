using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using SentimentAnalysis.ML.Base;
using SentimentAnalysis.ML.Objects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SentimentAnalysis.ML
{
    public class Trainer : BaseML
    {
        public void Train(string FileName)
        {
            if (!File.Exists(FileName))
            {
                Console.WriteLine("Failed to load the file");
                return;
            }

            // Load the text files into an IDataView object
            IDataView trainingDataView = MlContext.Data.LoadFromTextFile<SentimentData>(FileName);

            //Split data into test and training
            DataOperationsCatalog.TrainTestData dataSplit = MlContext.Data.TrainTestSplit(trainingDataView, 0.2);

            //Create the pipeline
            TextFeaturizingEstimator dataProcessPipeline = MlContext.Transforms.Text.
                FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.Text));

            //Create the trainer
            SdcaLogisticRegressionBinaryTrainer sdcaLogisticRegressionBinaryTrainer =
                MlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(SentimentData.Label), featureColumnName: "Features");

            //Append the trainer to pipeline
            EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>>
                trainingPipeline = dataProcessPipeline.Append(sdcaLogisticRegressionBinaryTrainer);

            //Train the model
            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);

            //Save the Model
            MlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, ModelPath);

            // Preparing test data.
            IDataView testSetTransform = trainedModel.Transform(dataSplit.TestSet);

            //Evaluate the model
            CalibratedBinaryClassificationMetrics modelMetrics = 
                MlContext.BinaryClassification.Evaluate(testSetTransform, labelColumnName: nameof(SentimentData.Label), scoreColumnName: nameof(SentimentPrediction.Score));

            //Print the metrics
            Console.WriteLine($"Area Under Curve: {modelMetrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"Accuracy: {modelMetrics.Accuracy:P2}");
            Console.WriteLine($"F1Score: {modelMetrics.F1Score:P2}");
            Console.WriteLine($"Positive Recall: {modelMetrics.PositiveRecall:#.##}");
            Console.WriteLine($"Negative Recall: {modelMetrics.NegativeRecall:#.##}");
        }
    }
}
