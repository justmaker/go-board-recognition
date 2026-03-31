import 'dart:math';
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'stone_classifier_weights.dart';
import 'board_state.dart';

/// Pure-Dart CNN inference for stone classification.
/// Architecture: 4x (Conv2d→BN→ReLU→MaxPool) → Flatten → FC→ReLU→Dropout → FC
/// Input: 32×32×3 RGB (normalized to [-1, 1])
/// Output: 3 classes (black, empty, white) — sorted alphabetically by ImageFolder
class StoneClassifierCNN {
  // Use StoneClassifierWeights.xxx directly for weight access

  /// Classify a 32×32 patch from a cv.Mat image.
  /// Returns StoneColor.
  StoneColor classify(cv.Mat patch) {
    // Convert cv.Mat to normalized float array [C, H, W] in [-1, 1]
    final input = _matToTensor(patch);
    final output = _forward(input);
    // Classes: 0=black, 1=empty, 2=white (alphabetical order from ImageFolder)
    final maxIdx = _argmax(output);
    switch (maxIdx) {
      case 0:
        return StoneColor.black;
      case 2:
        return StoneColor.white;
      default:
        return StoneColor.empty;
    }
  }

  /// Classify with confidence scores.
  (StoneColor, double) classifyWithConfidence(cv.Mat patch) {
    final input = _matToTensor(patch);
    final output = _forward(input);
    final probs = _softmax(output);
    final maxIdx = _argmax(output);
    final color = switch (maxIdx) {
      0 => StoneColor.black,
      2 => StoneColor.white,
      _ => StoneColor.empty,
    };
    return (color, probs[maxIdx]);
  }

  // ============================================================
  // Tensor operations
  // ============================================================

  /// Convert cv.Mat (BGR, HWC) to float tensor [C=3, H=32, W=32], normalized to [-1,1]
  Float32List _matToTensor(cv.Mat mat) {
    // Resize to 32x32 if needed
    cv.Mat resized;
    if (mat.rows != 32 || mat.cols != 32) {
      resized = cv.resize(mat, (32, 32));
    } else {
      resized = mat;
    }

    final data = resized.data;
    final h = resized.rows;
    final w = resized.cols;
    final channels = resized.channels;

    // Output: CHW format, RGB order, normalized to [-1, 1]
    final tensor = Float32List(3 * 32 * 32);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final idx = (y * w + x) * channels;
        // BGR → RGB, normalize: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
        tensor[0 * 1024 + y * 32 + x] = data[idx + 2] / 127.5 - 1.0; // R
        tensor[1 * 1024 + y * 32 + x] = data[idx + 1] / 127.5 - 1.0; // G
        tensor[2 * 1024 + y * 32 + x] = data[idx + 0] / 127.5 - 1.0; // B
      }
    }

    if (resized != mat) resized.dispose();
    return tensor;
  }

  /// Full forward pass
  List<double> _forward(Float32List input) {
    // Input: [3, 32, 32]
    var x = _conv2dBnReluPool(input, 3, 32, 32, 16,
        StoneClassifierWeights.features_0_weight, StoneClassifierWeights.features_0_bias,
        StoneClassifierWeights.features_1_weight, StoneClassifierWeights.features_1_bias,
        StoneClassifierWeights.features_1_running_mean, StoneClassifierWeights.features_1_running_var);
    // After pool: [16, 16, 16]

    x = _conv2dBnReluPool(x, 16, 16, 16, 32,
        StoneClassifierWeights.features_4_weight, StoneClassifierWeights.features_4_bias,
        StoneClassifierWeights.features_5_weight, StoneClassifierWeights.features_5_bias,
        StoneClassifierWeights.features_5_running_mean, StoneClassifierWeights.features_5_running_var);
    // After pool: [32, 8, 8]

    x = _conv2dBnReluPool(x, 32, 8, 8, 64,
        StoneClassifierWeights.features_8_weight, StoneClassifierWeights.features_8_bias,
        StoneClassifierWeights.features_9_weight, StoneClassifierWeights.features_9_bias,
        StoneClassifierWeights.features_9_running_mean, StoneClassifierWeights.features_9_running_var);
    // After pool: [64, 4, 4]

    x = _conv2dBnReluPool(x, 64, 4, 4, 64,
        StoneClassifierWeights.features_12_weight, StoneClassifierWeights.features_12_bias,
        StoneClassifierWeights.features_13_weight, StoneClassifierWeights.features_13_bias,
        StoneClassifierWeights.features_13_running_mean, StoneClassifierWeights.features_13_running_var);
    // After pool: [64, 2, 2] = 256

    // Flatten: already flat (x is [256])
    // FC1: 256 → 64
    var fc1 = _linear(x, 256, 64,
        StoneClassifierWeights.classifier_1_weight, StoneClassifierWeights.classifier_1_bias);
    fc1 = _relu(fc1);
    // Skip dropout (inference mode)

    // FC2: 64 → 3
    final fc2 = _linear(fc1, 64, 3,
        StoneClassifierWeights.classifier_4_weight, StoneClassifierWeights.classifier_4_bias);

    return fc2;
  }

  // ============================================================
  // Layer implementations
  // ============================================================

  /// Conv2d (3×3, padding=1) → BatchNorm → ReLU → MaxPool(2)
  Float32List _conv2dBnReluPool(
    Float32List input,
    int inC, int inH, int inW, int outC,
    List<double> convW, List<double> convB,
    List<double> bnW, List<double> bnB,
    List<double> bnMean, List<double> bnVar,
  ) {
    // Conv2d 3×3 padding=1
    final convOut = Float32List(outC * inH * inW);
    for (int oc = 0; oc < outC; oc++) {
      for (int oh = 0; oh < inH; oh++) {
        for (int ow = 0; ow < inW; ow++) {
          double sum = convB[oc];
          for (int ic = 0; ic < inC; ic++) {
            for (int kh = 0; kh < 3; kh++) {
              for (int kw = 0; kw < 3; kw++) {
                final ih = oh + kh - 1;
                final iw = ow + kw - 1;
                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                  final wIdx = oc * inC * 9 + ic * 9 + kh * 3 + kw;
                  final iIdx = ic * inH * inW + ih * inW + iw;
                  sum += convW[wIdx] * input[iIdx];
                }
              }
            }
          }
          convOut[oc * inH * inW + oh * inW + ow] = sum;
        }
      }
    }

    // BatchNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
    const eps = 1e-5;
    for (int c = 0; c < outC; c++) {
      final scale = bnW[c] / sqrt(bnVar[c] + eps);
      final shift = bnB[c] - bnMean[c] * scale;
      for (int i = 0; i < inH * inW; i++) {
        final idx = c * inH * inW + i;
        convOut[idx] = convOut[idx] * scale + shift;
      }
    }

    // ReLU
    for (int i = 0; i < convOut.length; i++) {
      if (convOut[i] < 0) convOut[i] = 0;
    }

    // MaxPool 2×2
    final outH = inH ~/ 2;
    final outW = inW ~/ 2;
    final poolOut = Float32List(outC * outH * outW);
    for (int c = 0; c < outC; c++) {
      for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
          double maxVal = -1e30;
          for (int ph = 0; ph < 2; ph++) {
            for (int pw = 0; pw < 2; pw++) {
              final ih = oh * 2 + ph;
              final iw = ow * 2 + pw;
              final val = convOut[c * inH * inW + ih * inW + iw];
              if (val > maxVal) maxVal = val;
            }
          }
          poolOut[c * outH * outW + oh * outW + ow] = maxVal;
        }
      }
    }

    return poolOut;
  }

  /// Fully connected layer
  List<double> _linear(
    Float32List input, int inF, int outF,
    List<double> weight, List<double> bias,
  ) {
    final output = List<double>.filled(outF, 0);
    for (int o = 0; o < outF; o++) {
      double sum = bias[o];
      for (int i = 0; i < inF; i++) {
        sum += weight[o * inF + i] * input[i];
      }
      output[o] = sum;
    }
    return output;
  }

  Float32List _relu(List<double> input) {
    final out = Float32List(input.length);
    for (int i = 0; i < input.length; i++) {
      out[i] = input[i] > 0 ? input[i] : 0;
    }
    return out;
  }

  List<double> _softmax(List<double> input) {
    final maxVal = input.reduce(max);
    final exps = input.map((x) => exp(x - maxVal)).toList();
    final sum = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sum).toList();
  }

  int _argmax(List<double> input) {
    int idx = 0;
    double maxVal = input[0];
    for (int i = 1; i < input.length; i++) {
      if (input[i] > maxVal) {
        maxVal = input[i];
        idx = i;
      }
    }
    return idx;
  }
}
