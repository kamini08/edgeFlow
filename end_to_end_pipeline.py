"""EdgeFlow Complete End-to-End Pipeline

Integrates all EdgeFlow components into a unified pipeline that demonstrates
the full developer experience from DSL to deployment.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from traceability_system import (
    ProvenanceTracker, 
    TransformationType, 
    trace_transformation,
    register_artifact,
    export_session_report
)
from optimization_orchestrator import (
    OptimizationOrchestrator, 
    OptimizationConfig,
    OptimizationLevel,
    OptimizationStrategy
)
from interactive_validator import InteractiveValidator
from dynamic_device_profiles import get_profile_manager, get_device_profile
from deployment_orchestrator import (
    CrossPlatformDeployer,
    DeploymentConfig,
    DeploymentTarget
)
from integrated_error_system import (
    get_error_reporter,
    ErrorCategory,
    ValidationSeverity
)

logger = logging.getLogger(__name__)


class EdgeFlowPipeline:
    """Complete EdgeFlow pipeline orchestrator."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.tracker = ProvenanceTracker(session_id)
        self.validator = InteractiveValidator(self.tracker)
        self.optimizer = OptimizationOrchestrator(self.tracker)
        self.deployer = CrossPlatformDeployer(self.tracker)
        self.error_reporter = get_error_reporter()
        self.profile_manager = get_profile_manager()
        
        logger.info(f"🚀 EdgeFlow Pipeline initialized (session: {self.tracker.session_id})")
    
    def run_complete_pipeline(
        self,
        dsl_file: str,
        model_path: str,
        output_dir: str = "pipeline_output",
        deploy_targets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the complete EdgeFlow pipeline from DSL to deployment."""
        
        start_time = time.perf_counter()
        results = {
            "success": False,
            "pipeline_duration_ms": 0.0,
            "stages_completed": [],
            "artifacts_generated": [],
            "deployment_results": {},
            "errors": [],
            "warnings": [],
        }
        
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            with trace_transformation(
                TransformationType.OPTIMIZATION,
                "end_to_end_pipeline",
                f"Complete EdgeFlow pipeline: {Path(dsl_file).name}",
                parameters={
                    "dsl_file": dsl_file,
                    "model_path": model_path,
                    "deploy_targets": deploy_targets or [],
                },
            ) as ctx:
                
                # Stage 1: DSL Validation
                logger.info("📋 Stage 1: DSL Validation and Parsing")
                validation_result = self._validate_dsl(dsl_file)
                if not validation_result["success"]:
                    results["errors"].extend(validation_result["errors"])
                    return results
                
                results["stages_completed"].append("validation")
                config = validation_result["config"]
                
                # Stage 2: Device Profile Selection
                logger.info("🎯 Stage 2: Device Profile Analysis")
                device_profile = self._analyze_target_device(config)
                if not device_profile:
                    results["errors"].append("Could not determine target device profile")
                    return results
                
                results["stages_completed"].append("device_analysis")
                
                # Stage 3: Model Optimization
                logger.info("⚡ Stage 3: Model Optimization")
                optimization_result = self._optimize_model(model_path, config, device_profile, output_dir)
                if not optimization_result["success"]:
                    results["errors"].extend(optimization_result["errors"])
                    return results
                
                results["stages_completed"].append("optimization")
                results["artifacts_generated"].extend(optimization_result["artifacts"])
                optimized_model = optimization_result["optimized_model_path"]
                
                # Stage 4: Performance Validation
                logger.info("📊 Stage 4: Performance Validation")
                perf_result = self._validate_performance(optimized_model, config, device_profile)
                results["warnings"].extend(perf_result.get("warnings", []))
                results["stages_completed"].append("performance_validation")
                
                # Stage 5: Code Generation
                logger.info("🛠️  Stage 5: Target Code Generation")
                codegen_result = self._generate_target_code(optimized_model, config, output_dir)
                results["artifacts_generated"].extend(codegen_result["artifacts"])
                results["stages_completed"].append("code_generation")
                
                # Stage 6: Deployment (if requested)
                if deploy_targets:
                    logger.info("🚀 Stage 6: Multi-Platform Deployment")
                    deployment_results = self._deploy_to_targets(
                        optimized_model, config, deploy_targets, output_dir
                    )
                    results["deployment_results"] = deployment_results
                    results["stages_completed"].append("deployment")
                
                # Stage 7: Report Generation
                logger.info("📄 Stage 7: Report Generation")
                report_result = self._generate_reports(output_dir)
                results["artifacts_generated"].extend(report_result["artifacts"])
                results["stages_completed"].append("reporting")
                
                results["success"] = True
                
                # Add metrics to traceability
                ctx.add_metric("stages_completed", len(results["stages_completed"]))
                ctx.add_metric("artifacts_generated", len(results["artifacts_generated"]))
                ctx.add_metric("deployments_attempted", len(deploy_targets or []))
                
                logger.info("✅ EdgeFlow pipeline completed successfully!")
        
        except Exception as e:
            results["errors"].append(str(e))
            logger.error(f"❌ Pipeline failed: {e}")
        
        finally:
            results["pipeline_duration_ms"] = (time.perf_counter() - start_time) * 1000
        
        return results
    
    def _validate_dsl(self, dsl_file: str) -> Dict[str, Any]:
        """Validate DSL file and parse configuration."""
        result = {"success": False, "config": {}, "errors": []}
        
        try:
            # Validate DSL syntax and semantics
            validation_result = self.validator.validate_file(dsl_file, show_progress=False)
            
            if validation_result.has_errors():
                for msg in validation_result.messages:
                    if msg.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                        result["errors"].append(str(msg))
                return result
            
            # Parse configuration
            from parser import parse_ef
            config = parse_ef(dsl_file)
            
            # Register DSL file as artifact
            register_artifact(
                f"dsl_config_{Path(dsl_file).name}",
                "config",
                dsl_file,
                {"stage": "input", "validation_passed": True},
                "user",
            )
            
            result["success"] = True
            result["config"] = config
            
        except Exception as e:
            result["errors"].append(f"DSL validation failed: {e}")
        
        return result
    
    def _analyze_target_device(self, config: Dict[str, Any]) -> Optional[Any]:
        """Analyze target device and get profile."""
        target_device = config.get("target_device", "cpu")
        
        # Get device profile
        device_profile = get_device_profile(target_device)
        
        if not device_profile:
            # Try to auto-detect if no profile found
            logger.warning(f"No profile found for {target_device}, attempting auto-detection")
            device_profile = self.profile_manager.auto_detect_device()
        
        if device_profile:
            logger.info(f"Using device profile: {device_profile.name}")
            
            # Register device profile as artifact
            register_artifact(
                f"device_profile_{device_profile.device_id}",
                "profile",
                metadata={
                    "device_name": device_profile.name,
                    "category": device_profile.category.value,
                    "constraints": len(device_profile.constraints),
                },
                created_by="device_analyzer",
            )
        
        return device_profile
    
    def _optimize_model(
        self, 
        model_path: str, 
        config: Dict[str, Any], 
        device_profile: Any,
        output_dir: str
    ) -> Dict[str, Any]:
        """Optimize model for target device."""
        result = {"success": False, "optimized_model_path": "", "artifacts": [], "errors": []}
        
        try:
            # Create optimization configuration
            opt_config = self.optimizer.create_optimization_config(
                target_device=device_profile.device_id,
                optimization_level=OptimizationLevel.BALANCED,
                strategy=OptimizationStrategy.BALANCED,
                **config
            )
            
            # Run optimization
            opt_result = self.optimizer.optimize_model(model_path, opt_config, output_dir)
            
            if opt_result.success:
                result["success"] = True
                result["optimized_model_path"] = opt_result.optimized_model_path
                result["artifacts"] = [opt_result.optimized_model_path] + opt_result.intermediate_models
                
                logger.info(f"Model optimized: {opt_result.size_reduction_percent:.1f}% size reduction")
            else:
                result["errors"] = opt_result.errors
        
        except Exception as e:
            result["errors"].append(f"Optimization failed: {e}")
        
        return result
    
    def _validate_performance(
        self, 
        model_path: str, 
        config: Dict[str, Any], 
        device_profile: Any
    ) -> Dict[str, Any]:
        """Validate model performance against targets."""
        result = {"success": True, "warnings": []}
        
        try:
            # Get performance targets from config
            target_latency = config.get("target_latency_ms")
            target_size = config.get("target_size_mb")
            
            # Benchmark model
            from benchmarker import benchmark_latency, get_model_size
            
            actual_latency, _ = benchmark_latency(model_path)
            actual_size = get_model_size(model_path)
            
            # Check against targets
            if target_latency and actual_latency > target_latency:
                result["warnings"].append(
                    f"Latency {actual_latency:.1f}ms exceeds target {target_latency}ms"
                )
            
            if target_size and actual_size > target_size:
                result["warnings"].append(
                    f"Model size {actual_size:.1f}MB exceeds target {target_size}MB"
                )
            
            # Check against device constraints
            memory_constraint = device_profile.get_constraint("max_memory_mb")
            if memory_constraint and actual_size * 4 > memory_constraint.value:  # 4x for inference overhead
                result["warnings"].append(
                    f"Model may not fit in device memory: {actual_size * 4:.1f}MB > {memory_constraint.value}MB"
                )
        
        except Exception as e:
            logger.warning(f"Performance validation failed: {e}")
        
        return result
    
    def _generate_target_code(
        self, 
        model_path: str, 
        config: Dict[str, Any], 
        output_dir: str
    ) -> Dict[str, Any]:
        """Generate target-specific inference code."""
        result = {"artifacts": []}
        
        try:
            from code_generator import CodeGenerator
            from edgeflow_ast import create_program_from_dict
            from edgeflow_ir import IRBuilder
            
            # Create AST and IR
            program = create_program_from_dict(config)
            ir_builder = IRBuilder()
            ir_graph = ir_builder.build_from_config(config)
            
            # Generate code
            generator = CodeGenerator(program, ir_graph)
            
            # Generate Python inference code
            python_code = generator.generate_python_inference()
            python_file = os.path.join(output_dir, "inference.py")
            with open(python_file, 'w') as f:
                f.write(python_code)
            result["artifacts"].append(python_file)
            
            # Generate C++ code for embedded deployment
            cpp_code = generator.generate_ir_based_code("cpp")
            cpp_file = os.path.join(output_dir, "inference.cpp")
            with open(cpp_file, 'w') as f:
                f.write(cpp_code)
            result["artifacts"].append(cpp_file)
            
            logger.info(f"Generated inference code: {len(result['artifacts'])} files")
        
        except Exception as e:
            logger.warning(f"Code generation failed: {e}")
        
        return result
    
    def _deploy_to_targets(
        self, 
        model_path: str, 
        config: Dict[str, Any], 
        deploy_targets: List[str],
        output_dir: str
    ) -> Dict[str, Any]:
        """Deploy to specified target platforms."""
        deployment_results = {}
        
        # Create deployment configurations
        deployment_configs = []
        for target_name in deploy_targets:
            if target_name == "raspberry_pi":
                deployment_configs.append(DeploymentConfig(
                    target=DeploymentTarget.RASPBERRY_PI,
                    model_path=model_path,
                    target_host=config.get("raspberry_pi_host", "raspberrypi.local")
                ))
            elif target_name == "docker":
                deployment_configs.append(DeploymentConfig(
                    target=DeploymentTarget.DOCKER,
                    model_path=model_path
                ))
            elif target_name == "kubernetes":
                deployment_configs.append(DeploymentConfig(
                    target=DeploymentTarget.KUBERNETES,
                    model_path=model_path
                ))
        
        # Deploy to all targets
        if deployment_configs:
            results = self.deployer.deploy_multi_target(model_path, deployment_configs)
            
            for target, result in results.items():
                deployment_results[target.value] = {
                    "success": result.success,
                    "deployment_url": result.deployment_url,
                    "artifacts": result.artifacts,
                    "errors": result.errors,
                }
        
        return deployment_results
    
    def _generate_reports(self, output_dir: str) -> Dict[str, Any]:
        """Generate comprehensive reports."""
        result = {"artifacts": []}
        
        try:
            # Generate provenance report
            provenance_file = os.path.join(output_dir, "provenance_report.json")
            export_session_report(provenance_file)
            result["artifacts"].append(provenance_file)
            
            # Generate error report
            error_file = os.path.join(output_dir, "error_report.md")
            self.error_reporter.generate_error_report(error_file)
            result["artifacts"].append(error_file)
            
            # Generate summary report
            summary_file = os.path.join(output_dir, "pipeline_summary.json")
            summary = {
                "session_id": self.tracker.session_id,
                "pipeline_statistics": self.tracker.get_summary_statistics(),
                "error_summary": self.error_reporter.get_session_summary(),
                "artifacts_generated": len(self.tracker.artifacts),
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            result["artifacts"].append(summary_file)
            
            logger.info(f"Generated reports: {len(result['artifacts'])} files")
        
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")
        
        return result


def run_demo_pipeline() -> None:
    """Run a demonstration of the complete EdgeFlow pipeline."""
    
    # Create demo DSL file
    demo_dsl = """
# EdgeFlow Demo Configuration
model = "mobilenet_v2.tflite"
target_device = "raspberry_pi"
quantize = "int8"
optimize_for = "size"
target_latency_ms = 50
target_size_mb = 2.0
enable_fusion = true
deploy_raspberry_pi = true
deploy_docker = true
"""
    
    with open("demo_config.ef", "w") as f:
        f.write(demo_dsl)
    
    # Initialize pipeline
    pipeline = EdgeFlowPipeline()
    
    # Run complete pipeline
    logger.info("🚀 Starting EdgeFlow demo pipeline...")
    
    results = pipeline.run_complete_pipeline(
        dsl_file="demo_config.ef",
        model_path="mobilenet_v2.tflite",  # Would need actual model file
        deploy_targets=["docker"],  # Simplified for demo
    )
    
    # Print results
    print("\n" + "="*60)
    print("EdgeFlow Pipeline Results")
    print("="*60)
    print(f"Success: {results['success']}")
    print(f"Duration: {results['pipeline_duration_ms']:.1f}ms")
    print(f"Stages Completed: {', '.join(results['stages_completed'])}")
    print(f"Artifacts Generated: {len(results['artifacts_generated'])}")
    
    if results['errors']:
        print(f"Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")
    
    if results['warnings']:
        print(f"Warnings: {len(results['warnings'])}")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    print("="*60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_demo_pipeline()
