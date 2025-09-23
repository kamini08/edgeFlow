import React, { useState } from "react";
import Link from "next/link";
import FileUpload from "../components/FileUpload";
import ConfigEditor from "../components/ConfigEditor";
import ASTViewer from "../components/ASTViewer";
import IRGraphViewer from "../components/IRGraphViewer";
import EnhancedASTViewer from "../components/EnhancedASTViewer";
import EnhancedIRGraphViewer from "../components/EnhancedIRGraphViewer";
import ValidationResultsViewer from "../components/ValidationResultsViewer";
import DeploymentArtifactsViewer from "../components/DeploymentArtifactsViewer";
import OptimizationPassesViewer from "../components/OptimizationPassesViewer";
import GeneratedCodeViewer from "../components/GeneratedCodeViewer";
import {
  compileConfig,
  compileConfigVerbose,
  runFullPipeline,
  fastCompile,
  validateDeployment,
  packageDeployment,
  deviceBenchmark,
  benchmarkInterfaces,
} from "../services/api";

export default function CompilePage() {
  const [content, setContent] = useState("");
  const [filename, setFilename] = useState("config.ef");
  const [logs, setLogs] = useState<string[]>([]);
  const [parsed, setParsed] = useState<any>(null);
  const [pipelineResults, setPipelineResults] = useState<any>(null);
  const [validationResults, setValidationResults] = useState<any>(null);
  const [deploymentArtifacts, setDeploymentArtifacts] = useState<any>(null);
  const [fastCompileResults, setFastCompileResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const onFileSelect = (file: File) => {
    setFilename(file.name);
    file.text().then(setContent);
  };

  const onCompile = async (verbose = false) => {
    setLoading(true);
    try {
      const fn = verbose ? compileConfigVerbose : compileConfig;
      const res = await fn(content, filename);
      setParsed(res.parsed_config);
      setLogs(res.logs || []);
    } catch (error) {
      console.error("Compile error:", error);
    } finally {
      setLoading(false);
    }
  };

  const onRunPipeline = async () => {
    setLoading(true);
    try {
      const res = await runFullPipeline(content, filename);
      setPipelineResults(res);
    } catch (error) {
      console.error("Pipeline error:", error);
    } finally {
      setLoading(false);
    }
  };

  const onFastCompile = async () => {
    setLoading(true);
    try {
      const res = await fastCompile(content, filename);
      setFastCompileResults(res);
    } catch (error) {
      console.error("Fast compile error:", error);
    } finally {
      setLoading(false);
    }
  };

  const onPackageDeployment = async () => {
    setLoading(true);
    try {
      const res = await packageDeployment(content, filename);
      setDeploymentArtifacts(res);
    } catch (error) {
      console.error("Deployment packaging error:", error);
    } finally {
      setLoading(false);
    }
  };

  const onValidateDeployment = async (
    packagePath: string,
    deviceType: string
  ) => {
    setLoading(true);
    try {
      const res = await validateDeployment(packagePath, deviceType);
      setValidationResults(res);
    } catch (error) {
      console.error("Deployment validation error:", error);
    } finally {
      setLoading(false);
    }
  };

  const onDeviceBenchmark = async () => {
    setLoading(true);
    try {
      const res = await deviceBenchmark(content, filename);
      setPipelineResults((prev) => ({ ...prev, benchmarkResults: res }));
    } catch (error) {
      console.error("Device benchmark error:", error);
    } finally {
      setLoading(false);
    }
  };

  const onBenchmarkInterfaces = async () => {
    setLoading(true);
    try {
      const res = await benchmarkInterfaces(content, filename);
      setPipelineResults((prev) => ({
        ...prev,
        interfaceBenchmarkResults: res,
      }));
    } catch (error) {
      console.error("Interface benchmark error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b bg-white">
        <div className="container-narrow flex items-center justify-between py-4">
          <h1 className="text-xl font-semibold text-gray-900">EdgeFlow</h1>
          <nav className="space-x-4 text-sm font-medium">
            <Link className="text-gray-700 hover:text-blue-600" href="/">
              Home
            </Link>
            <Link className="text-gray-700 hover:text-blue-600" href="/results">
              Results
            </Link>
          </nav>
        </div>
      </header>
      <main className="container-narrow py-10 space-y-6">
        <section className="card space-y-4">
          <h2 className="text-lg font-semibold text-gray-900">
            Compile Config
          </h2>
          <div>
            <FileUpload
              onFileSelect={onFileSelect}
              acceptedFormats={[".ef"]}
              maxSize={5}
            />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-700">
              {filename}
            </label>
            <ConfigEditor initialContent={content} onChange={setContent} />
          </div>
          <div className="flex gap-3 flex-wrap">
            <button
              className="btn"
              onClick={() => onCompile(false)}
              disabled={loading}>
              {loading ? "Compiling..." : "Compile"}
            </button>
            <button
              className="btn bg-gray-700 hover:bg-gray-800"
              onClick={() => onCompile(true)}
              disabled={loading}>
              {loading ? "Compiling..." : "Compile Verbose"}
            </button>
            <button
              className="btn bg-blue-600 hover:bg-blue-700 text-white"
              onClick={onRunPipeline}
              disabled={loading}>
              {loading ? "Running..." : "Run Full Pipeline"}
            </button>
            <button
              className="btn bg-green-600 hover:bg-green-700 text-white"
              onClick={onFastCompile}
              disabled={loading}>
              {loading ? "Analyzing..." : "Fast Compile"}
            </button>
            <button
              className="btn bg-purple-600 hover:bg-purple-700 text-white"
              onClick={onPackageDeployment}
              disabled={loading}>
              {loading ? "Packaging..." : "Package Deployment"}
            </button>
            <button
              className="btn bg-orange-600 hover:bg-orange-700 text-white"
              onClick={onDeviceBenchmark}
              disabled={loading}>
              {loading ? "Benchmarking..." : "Device Benchmark"}
            </button>
            <button
              className="btn bg-indigo-600 hover:bg-indigo-700 text-white"
              onClick={onBenchmarkInterfaces}
              disabled={loading}>
              {loading ? "Testing..." : "Benchmark Interfaces"}
            </button>
          </div>
        </section>
        {parsed && (
          <section className="card">
            <h3 className="mb-2 text-base font-semibold text-gray-900">
              Parsed Config
            </h3>
            <pre className="overflow-auto whitespace-pre-wrap text-sm text-gray-800">
              {JSON.stringify(parsed, null, 2)}
            </pre>
          </section>
        )}
        {logs.length > 0 && (
          <section className="card">
            <h3 className="mb-2 text-base font-semibold text-gray-900">Logs</h3>
            <pre className="overflow-auto whitespace-pre-wrap text-sm text-gray-800">
              {logs.join("\n")}
            </pre>
          </section>
        )}

        {/* Fast Compile Results */}
        {fastCompileResults && (
          <section className="card">
            <h3 className="mb-4 text-lg font-semibold text-gray-900">
              Fast Compile Analysis
            </h3>
            {fastCompileResults.success ? (
              <div className="space-y-4">
                {fastCompileResults.estimated_impact && (
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <h4 className="font-semibold text-green-900 mb-2">
                      Estimated Impact
                    </h4>
                    <pre className="text-sm text-green-800">
                      {JSON.stringify(
                        fastCompileResults.estimated_impact,
                        null,
                        2
                      )}
                    </pre>
                  </div>
                )}
                {fastCompileResults.validation_results && (
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-900 mb-2">
                      Validation Results
                    </h4>
                    <pre className="text-sm text-blue-800">
                      {JSON.stringify(
                        fastCompileResults.validation_results,
                        null,
                        2
                      )}
                    </pre>
                  </div>
                )}
                {fastCompileResults.warnings &&
                  fastCompileResults.warnings.length > 0 && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                      <h4 className="font-semibold text-yellow-900 mb-2">
                        Warnings
                      </h4>
                      <ul className="text-sm text-yellow-800">
                        {fastCompileResults.warnings.map(
                          (warning: string, index: number) => (
                            <li key={index}>• {warning}</li>
                          )
                        )}
                      </ul>
                    </div>
                  )}
              </div>
            ) : (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-red-800">{fastCompileResults.message}</p>
              </div>
            )}
          </section>
        )}

        {/* Pipeline Results */}
        {pipelineResults && (
          <div className="space-y-6">
            {pipelineResults.success ? (
              <>
                {/* Enhanced AST Viewer */}
                {pipelineResults.ast && (
                  <EnhancedASTViewer ast={pipelineResults.ast} />
                )}

                {/* Enhanced IR Graph Viewer */}
                {pipelineResults.ir_graph && (
                  <EnhancedIRGraphViewer irGraph={pipelineResults.ir_graph} />
                )}

                {/* Optimization Passes */}
                {pipelineResults.optimization_passes && (
                  <OptimizationPassesViewer
                    optimizationPasses={pipelineResults.optimization_passes}
                  />
                )}

                {/* Generated Code */}
                {pipelineResults.generated_code && (
                  <GeneratedCodeViewer
                    generatedCode={pipelineResults.generated_code}
                  />
                )}

                {/* Explainability Report */}
                {pipelineResults.explainability_report && (
                  <section className="card">
                    <h3 className="mb-4 text-lg font-semibold text-gray-900">
                      Explainability Report
                    </h3>
                    <div className="prose max-w-none">
                      <pre className="whitespace-pre-wrap text-sm text-gray-800 bg-gray-50 p-4 rounded-lg">
                        {pipelineResults.explainability_report}
                      </pre>
                    </div>
                  </section>
                )}

                {/* Benchmark Results */}
                {pipelineResults.benchmarkResults && (
                  <section className="card">
                    <h3 className="mb-4 text-lg font-semibold text-gray-900">
                      Device Benchmark Results
                    </h3>
                    <div className="prose max-w-none">
                      <pre className="whitespace-pre-wrap text-sm text-gray-800 bg-gray-50 p-4 rounded-lg">
                        {JSON.stringify(
                          pipelineResults.benchmarkResults,
                          null,
                          2
                        )}
                      </pre>
                    </div>
                  </section>
                )}

                {/* Interface Benchmark Results */}
                {pipelineResults.interfaceBenchmarkResults && (
                  <section className="card">
                    <h3 className="mb-4 text-lg font-semibold text-gray-900">
                      Interface Benchmark Results
                    </h3>
                    <div className="prose max-w-none">
                      <pre className="whitespace-pre-wrap text-sm text-gray-800 bg-gray-50 p-4 rounded-lg">
                        {JSON.stringify(
                          pipelineResults.interfaceBenchmarkResults,
                          null,
                          2
                        )}
                      </pre>
                    </div>
                  </section>
                )}
              </>
            ) : (
              <section className="card">
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <h3 className="font-semibold text-red-900 mb-2">
                    Pipeline Failed
                  </h3>
                  {pipelineResults.errors && (
                    <ul className="text-red-800">
                      {pipelineResults.errors.map(
                        (error: string, index: number) => (
                          <li key={index}>• {error}</li>
                        )
                      )}
                    </ul>
                  )}
                </div>
              </section>
            )}
          </div>
        )}

        {/* Deployment Artifacts */}
        {deploymentArtifacts && (
          <div className="space-y-6">
            {deploymentArtifacts.success ? (
              <>
                <DeploymentArtifactsViewer
                  artifacts={deploymentArtifacts.artifacts}
                  deviceType={deploymentArtifacts.device_type || "unknown"}
                />

                {/* Validation Button */}
                {deploymentArtifacts.final_package && (
                  <section className="card">
                    <h3 className="mb-4 text-lg font-semibold text-gray-900">
                      Validate Deployment Package
                    </h3>
                    <div className="flex gap-3">
                      <button
                        className="btn bg-red-600 hover:bg-red-700 text-white"
                        onClick={() =>
                          onValidateDeployment(
                            deploymentArtifacts.final_package,
                            deploymentArtifacts.device_type || "raspberry_pi"
                          )
                        }
                        disabled={loading}>
                        {loading ? "Validating..." : "Validate Package"}
                      </button>
                    </div>
                  </section>
                )}
              </>
            ) : (
              <section className="card">
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <h3 className="font-semibold text-red-900 mb-2">
                    Deployment Packaging Failed
                  </h3>
                  <p className="text-red-800">{deploymentArtifacts.message}</p>
                </div>
              </section>
            )}
          </div>
        )}

        {/* Validation Results */}
        {validationResults && (
          <ValidationResultsViewer validationReport={validationResults} />
        )}
      </main>
    </div>
  );
}
