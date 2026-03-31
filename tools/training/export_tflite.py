"""
Export trained PyTorch model to TFLite format.
PyTorch → ONNX → TFLite
"""

import os
import sys
import numpy as np


def export_tflite(pth_path: str, output_path: str):
    import torch
    from train import StoneCNN

    # Load PyTorch model
    model = StoneCNN(3)
    model.load_state_dict(torch.load(pth_path, map_location='cpu', weights_only=True))
    model.eval()

    # Export to ONNX first
    onnx_path = output_path.replace('.tflite', '.onnx')
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy, onnx_path,
        input_names=['input'], output_names=['output'],
        opset_version=13)
    print(f"ONNX: {os.path.getsize(onnx_path) / 1024:.1f} KB")

    # Convert ONNX → TFLite via TensorFlow
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)

        # Save as SavedModel
        saved_model_dir = output_path.replace('.tflite', '_savedmodel')
        tf_rep.export_graph(saved_model_dir)

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite: {len(tflite_model) / 1024:.1f} KB → {output_path}")
        return True

    except ImportError:
        print("onnx-tf not available, trying alternative method...")

    # Alternative: use tf2onnx approach or just keep ONNX
    try:
        import subprocess
        # Try using onnx2tf CLI if available
        result = subprocess.run(
            ['onnx2tf', '-i', onnx_path, '-o', output_path.replace('.tflite', '_tf'),
             '-oiqt', '-qt', 'per-tensor'],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            # Find the tflite file
            tf_dir = output_path.replace('.tflite', '_tf')
            for f in os.listdir(tf_dir):
                if f.endswith('.tflite'):
                    import shutil
                    shutil.copy2(os.path.join(tf_dir, f), output_path)
                    print(f"TFLite: {os.path.getsize(output_path) / 1024:.1f} KB → {output_path}")
                    return True
        print(f"onnx2tf failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"Alternative conversion failed: {e}")

    # Fallback: just keep ONNX for now
    print(f"\nCould not convert to TFLite. ONNX model available at: {onnx_path}")
    print("For Flutter integration, consider using onnxruntime_flutter instead of tflite_flutter.")
    return False


if __name__ == '__main__':
    pth_path = 'output/model/stone_classifier.pth'
    output_path = 'output/model/stone_classifier.tflite'

    if not os.path.exists(pth_path):
        print(f"Model not found: {pth_path}")
        sys.exit(1)

    export_tflite(pth_path, output_path)
