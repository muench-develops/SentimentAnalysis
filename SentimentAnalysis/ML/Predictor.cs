using Microsoft.ML;
using SentimentAnalysis.ML.Base;
using SentimentAnalysis.ML.Objects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SentimentAnalysis.ML
{
    public class Predictor : BaseML
    {
        public void Predict(string inputData)
        {
            if(!File.Exists(ModelPath))
            {
                Console.WriteLine("Failed to load model");
                return;
            }

            // Loading the model
            ITransformer mlModel;
            using var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            mlModel = MlContext.Model.Load(stream, out _);

            if(mlModel != null)
            {
                //Create the predictionEngine
                var predictionEngine = MlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlModel);
                //Make the prediction
                var prediction = predictionEngine.Predict(new SentimentData { Text = inputData });
                // Output result
                Console.WriteLine($"Based on {inputData} - the feedback is: {(prediction.Prediction ? "Positive" : "Negative")} with Probability {prediction.Probability:P0}");
            }
            else
            {
                Console.WriteLine("The model couldn't be loaded");
                return;
            }

        }
    }
}
