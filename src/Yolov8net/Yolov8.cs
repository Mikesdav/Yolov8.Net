using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
using System.Drawing;
using Yolov8Net.Extentions;

namespace Yolov8Net
{
    public class YoloV8Predictor
        : PredictorBase, IPredictor
    {
        public static IPredictor Create(string modelPath, string[]? labels = null, bool useCuda = false)
        {
            return new YoloV8Predictor(modelPath, labels, useCuda);
        }

        private YoloV8Predictor(string modelPath, string[]? labels = null, bool useCuda = false)
            : base(modelPath, labels, useCuda) { }

        protected List<Prediction> ParseOutput(DenseTensor<float> output, Image image)
        {
            var result = new List<Prediction>((int)(output.Length / output.Dimensions[1]));
            int labelsCount = ModelOutputDimensions - 4;

            var (w, h) = (image.Width, image.Height);
            var (xGain, yGain) = (ModelInputWidth / (float)w, ModelInputHeight / (float)h);
            var gain = Math.Min(xGain, yGain);
            var (xPad, yPad) = ((ModelInputWidth - w * gain) / 2, (ModelInputHeight - h * gain) / 2);

            for (int i = 0; i < output.Dimensions[0]; i++)
            {
                for (int j = 0; j < output.Length / output.Dimensions[1]; j++)
                {
                    float xMin = ((output[i, 0, j] - output[i, 2, j] / 2) - xPad) / gain;
                    float yMin = ((output[i, 1, j] - output[i, 3, j] / 2) - yPad) / gain;
                    float xMax = ((output[i, 0, j] + output[i, 2, j] / 2) - xPad) / gain;
                    float yMax = ((output[i, 1, j] + output[i, 3, j] / 2) - yPad) / gain;

                    xMin = Utils.Clamp(xMin, 0, w - 0);
                    yMin = Utils.Clamp(yMin, 0, h - 0);
                    xMax = Utils.Clamp(xMax, 0, w - 1);
                    yMax = Utils.Clamp(yMax, 0, h - 1);

                    for (int l = 0; l < labelsCount; l++)
                    {
                        var pred = output[i, 4 + l, j];
                        if (pred >= Confidence)
                        {
                            result.Add(new Prediction()
                            {
                                Label = Labels[l],
                                Score = pred,
                                Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                            });
                        }
                    }
                }
            }

            return result;
        }
        override public Prediction[] Predict(Image image)
        {
            return Suppress(
                ParseOutput(
                    Inference(image)[0], image)
                );
        }
    }
}
