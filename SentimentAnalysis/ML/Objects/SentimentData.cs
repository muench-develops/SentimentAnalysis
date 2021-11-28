using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SentimentAnalysis.ML.Objects
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string? Text;

        [LoadColumn(1)]
        public bool Label;
    }
}
