import unittest

from mlir_dialect import MLIRModule
from pipeline import compile_model
from unified_ir import UIRGraph


class TestHighLevelPipeline(unittest.TestCase):
    def test_compile_model_simulated_tf(self):
        mlir_module, final_graph, validation = compile_model(
            model_path="test_model.h5", target_device="cpu", quantize="int8"
        )
        self.assertIsInstance(mlir_module, MLIRModule)
        self.assertIsInstance(final_graph, UIRGraph)
        self.assertIsNotNone(validation)

    def test_compile_model_simulated_onnx(self):
        mlir_module, final_graph, validation = compile_model(
            model_path="test_model.onnx",
            target_device="jetson_nano",
            quantize="float16",
        )
        self.assertIsInstance(mlir_module, MLIRModule)
        self.assertIsInstance(final_graph, UIRGraph)
        self.assertIsNotNone(validation)


if __name__ == "__main__":
    unittest.main()
