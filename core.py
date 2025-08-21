# =============================================================
# Standard Library Imports
# =============================================================
import os
import json
import importlib

# =============================================================
# Third-Party Libraries
# =============================================================
import tensorflow as tf
import tensorflow_transform as tft

# =============================================================
# TFX Core Libraries
# =============================================================
from tfx.types import (
                        artifact_utils,
                        channel_utils,
                        standard_artifacts,
                        component_spec
                    )
from tfx.types.component_spec import ChannelParameter, ExecutionParameter

from tfx.dsl.components.base import (
                                        base_component,
                                        executor_spec,
                                        base_executor
                                    )




# =============================================================
# [1] MultiOutputEvaluator Component Spec
# =============================================================
class MultiOutputEvaluatorSpec(component_spec.ComponentSpec):
    """
    Defines the specification (contract) for the MultiOutputEvaluator component.

    This spec declares:
    - PARAMETERS: Execution-time arguments (hyperparameters, configurations).
    - INPUTS: Input artifacts (Model, Examples, TransformGraph).
    - OUTPUTS: Output artifacts (ModelEvaluation).

    Attributes:
        output_names (list): List of model output names (e.g., ["y1", "y2"]).
        metrics (list): List of metrics to compute (e.g., ["mse", "mae"]).
        example_split (str): Which data split to evaluate (default: 'test').
        input_fn_path (str): Dotted path to input function for dataset creation.
        input_fn_kwargs (dict, optional): Extra kwargs for the input function.
    """
    
    PARAMETERS = {
        'output_names': ExecutionParameter(type=list),                              # List of output names
        'metrics': ExecutionParameter(type=list),                                   # List of metrics to compute

        'example_split': ExecutionParameter(type=str),                              # Example: 'test'
        'input_fn_path': ExecutionParameter(type=str),                              # format: "my_pkg.my_input_module:make_dataset" or "my_pkg.my_input_module.make_dataset"
        'input_fn_kwargs': ExecutionParameter(type=dict, optional=True),            # JSON-serializable kwargs for the input_fn
    }
    
    INPUTS = {
        'model': ChannelParameter(type=standard_artifacts.Model),                   # Model artifact
        'examples': ChannelParameter(type=standard_artifacts.Examples),             # Example artifact
        'transform_graph': ChannelParameter(type=standard_artifacts.TransformGraph) # TransformGraph artifact
    }
    
    OUTPUTS = {
        'evaluation': ChannelParameter(type=standard_artifacts.ModelEvaluation)     # ModelEvaluation artifact
    }




# =============================================================
# [2] MultiOutputEvaluator Executor
# =============================================================
class Executor(base_executor.BaseExecutor):
    """
    Executor for MultiOutputEvaluator.

    This class defines the actual logic of the component:
    - Loads the trained model, transform graph, and evaluation dataset.
    - Computes per-output and global metrics.
    - Writes evaluation results in a TFMA-compatible JSON format.
    """
    
    # ---------------------------------------------------------
    # Utility: Dynamically import a Python object from string path
    # ---------------------------------------------------------
    def _load_obj_from_dotted_path(self, dotted: str):
        """
        Load an object given a dotted path.

        Supports formats:
        - "pkg.module:object"
        - "pkg.module.object"
        """
        if ':' in dotted:
            mod_path, obj_name = dotted.split(':', 1)
        else:
            parts = dotted.split('.')
            mod_path, obj_name = '.'.join(parts[:-1]), parts[-1]
        module = importlib.import_module(mod_path)
        return getattr(module, obj_name)

    # ---------------------------------------------------------
    # Core execution method
    # ---------------------------------------------------------
    def Do(self, input_dict, output_dict, exec_properties):
        """
        Compute evaluation metrics for multiple model outputs.

        Args:
            input_dict (dict): Mapping of input keys to artifact lists.
            output_dict (dict): Mapping of output keys to artifact lists.
            exec_properties (dict): Runtime execution parameters.
        """
        
        # -----------------------------------------------------
        # Step 1: Resolve artifact URIs
        # -----------------------------------------------------
        model_path = os.path.join(artifact_utils.get_single_uri(input_dict['model']), 'Format-Serving')
        examples_uri = artifact_utils.get_single_uri(input_dict['examples'])
        transform_graph_uri = artifact_utils.get_single_uri(input_dict['transform_graph'])
        

        # -----------------------------------------------------
        # Step 2: Load transform graph and model
        # -----------------------------------------------------
        tf_transform_output = tft.TFTransformOutput(transform_graph_uri)
        model = tf.saved_model.load(model_path)
        

        # -----------------------------------------------------
        # Step 3: Extract execution properties
        # -----------------------------------------------------
        input_fn_path = exec_properties['input_fn_path']  # required
        input_fn_kwargs = exec_properties.get('input_fn_kwargs', {}) or {}
        output_names = exec_properties['output_names']
        metrics = exec_properties['metrics']
        example_split = exec_properties.get('example_split', 'test')
        

        # -----------------------------------------------------
        # Step 4: Build dataset using provided input function
        # -----------------------------------------------------
        file_pattern = tf.io.gfile.glob(os.path.join(examples_uri, example_split, '*.gz'))
        input_fn = self._load_obj_from_dotted_path(input_fn_path)
        dataset = input_fn(file_pattern, tf_transform_output, **input_fn_kwargs)
        
        
        # -----------------------------------------------------
        # Step 5: Initialize metrics (per-output and global)
        # -----------------------------------------------------
        per_output_metrics = {}
        for output_name in output_names:
            per_output_metrics[output_name] = {}
            for metric_name in metrics:
                if metric_name == 'mse':
                    per_output_metrics[output_name][metric_name] = tf.keras.metrics.MeanSquaredError()
                elif metric_name == 'mae':
                    per_output_metrics[output_name][metric_name] = tf.keras.metrics.MeanAbsoluteError()
        
        global_metrics = {}
        for metric_name in metrics:
            if metric_name == 'mse':
                global_metrics[metric_name] = tf.keras.metrics.MeanSquaredError()
            elif metric_name == 'mae':
                global_metrics[metric_name] = tf.keras.metrics.MeanAbsoluteError()
        
        
        # -----------------------------------------------------
        # Step 6: Evaluate model predictions
        # -----------------------------------------------------
        for features, labels in dataset:
            
            predictions = model(features)

            # Per-output updates
            for i, output_name in enumerate(output_names):
                labels_slice = labels[:, 0, i]  # Shape: (batch_size,)
                output_pred = predictions[:, i]  # Shape: (batch_size,)
                for metric_name, metric in per_output_metrics[output_name].items():
                    metric.update_state(labels_slice, output_pred)
            
            # Global updates (flatten all outputs)
            predictions_flat = tf.reshape(predictions, [-1])
            labels_flat = tf.reshape(labels, [-1])
            for metric_name, metric in global_metrics.items():
                metric.update_state(labels_flat, predictions_flat)


        # -----------------------------------------------------
        # Step 7: Collect final metric results
        # -----------------------------------------------------
        results = {
                    'per_output': {},
                    'global': {}
                }
        
        # 7.1. Per-output results
        for output_name in output_names:
            results['per_output'][output_name] = {}
            for metric_name, metric in per_output_metrics[output_name].items():
                results['per_output'][output_name][metric_name] = float(metric.result().numpy())
        
        # 7.2. Global results
        for metric_name, metric in global_metrics.items():
            results['global'][metric_name] = float(metric.result().numpy())
        
        
        # -----------------------------------------------------
        # Step 8: Write results to output artifact
        # -----------------------------------------------------
        evaluation = artifact_utils.get_single_instance(output_dict['evaluation'])
        tfma_results = self._format_as_tfma_results(results, output_names)
        
        
        # Save as JSON file
        evaluation.uri = os.path.join(evaluation.uri, "evaluation.json")  # Append a filename
        tf.io.gfile.makedirs(os.path.dirname(evaluation.uri))
        with tf.io.gfile.GFile(evaluation.uri, 'w') as f:
            f.write(json.dumps(tfma_results))


    # ---------------------------------------------------------
    # Helper: Format results for TFMA compatibility
    # ---------------------------------------------------------
    def _format_as_tfma_results(self, results, output_names):
        """
        Convert raw metric results into TFMA-compatible JSON format.

        Args:
            results (dict): Computed metrics.
            output_names (list): Names of model outputs.

        Returns:
            dict: TFMA-compatible results.
        """
        # TFMA Results
        tfma_results = {
                        'version': '0.0.1',
                        'model_specs': [{
                            'name': output_name,
                            'is_baseline': False
                        } for output_name in output_names],
                        'metrics': {}
                    }

        # Record per-output metrics
        mse_values = []
        mae_values = []
        for output_name in output_names:
            for metric_name, value in results['per_output'][output_name].items():
                metric_key = f'per_output>>{output_name}>>{metric_name}'
                tfma_results['metrics'][metric_key] = value

                # Store all MSE and MAE values
                if metric_name.lower() == "mse":
                    mse_values.append((output_name, value))
                elif metric_name.lower() == "mae":
                    mae_values.append((output_name, value))
            
        # Record global metrics
        for metric_name, value in results['global'].items():
            metric_key = f'global>>{metric_name}'
            tfma_results['metrics'][metric_key] = value
        
        return tfma_results




# =============================================================
# [3] MultiOutputEvaluator Component
# =============================================================
class MultiOutputEvaluator(base_component.BaseComponent):
    """
    A custom TFX component for evaluating models with multiple outputs.

    This component:
    - Accepts a trained model, examples, and transform graph.
    - Evaluates the model across multiple outputs using specified metrics.
    - Produces a ModelEvaluation artifact (TFMA-compatible JSON file).
    """
    
    SPEC_CLASS = MultiOutputEvaluatorSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)
    
    def __init__(self, 
                model,
                examples,
                transform_graph,

                output_names,
                metrics,

                input_fn_path,
                input_fn_kwargs=None,

                example_split='test'):
        """
        Initialize MultiOutputEvaluator.

        Args:
            model (Channel): Trained model artifact.
            examples (Channel): Examples artifact (train/test data).
            transform_graph (Channel): TransformGraph artifact.
            output_names (list): Names of outputs to evaluate.
            metrics (list): Metrics to compute (e.g., ["mse", "mae"]).
            input_fn_path (str): Path to dataset input function.
            input_fn_kwargs (dict, optional): Arguments for dataset function.
            example_split (str): Data split to evaluate (default: "test").
        """
        evaluation = channel_utils.as_channel([standard_artifacts.ModelEvaluation()])
        
        spec = MultiOutputEvaluatorSpec(
            model=model,
            examples=examples,
            transform_graph=transform_graph,
            output_names=output_names,
            
            metrics=metrics,

            input_fn_path=input_fn_path,
            input_fn_kwargs=input_fn_kwargs or {},

            example_split=example_split,
            evaluation=evaluation  # Only pass evaluation, not metrics_artifact
        )
        
        super(MultiOutputEvaluator, self).__init__(spec=spec)