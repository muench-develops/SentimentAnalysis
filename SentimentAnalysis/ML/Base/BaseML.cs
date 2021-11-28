using Microsoft.ML;
using SentimentAnalysis.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SentimentAnalysis.ML.Base
{
    public class BaseML
    {
        protected readonly MLContext MlContext;
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, Constants.MODEL_FILENAME);

        protected BaseML()
        {
            MlContext = new MLContext(2020);
        }
    }
}
