// See https://aka.ms/new-console-template for more information
using SentimentAnalysis.ML;

Console.WriteLine("Hello, ML.NET World!");
Console.WriteLine("Please Enter the path to train: (empty = only predict)");

string path = Console.ReadLine();

if (!string.IsNullOrEmpty(path))
    new Trainer().Train(FileName: path);

Console.WriteLine("Enter sentence to predict: ");

string predictSentence = Console.ReadLine();
if (!string.IsNullOrEmpty(predictSentence))
    new Predictor().Predict(predictSentence);


Console.ReadKey();