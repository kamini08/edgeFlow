import React, { useState } from "react";
import Link from "next/link";
import FileUpload from "../components/FileUpload";
import ConfigEditor from "../components/ConfigEditor";
import ASTViewer from "../components/ASTViewer";
import IRGraphViewer from "../components/IRGraphViewer";
import OptimizationPassesViewer from "../components/OptimizationPassesViewer";
import GeneratedCodeViewer from "../components/GeneratedCodeViewer";
import {
  compileConfig,
  compileConfigVerbose,
  runFullPipeline,
  fastCompile,
} from "../services/api";

export default function CompilePage() {
  const [content, setContent] = useState("");
  const [filename, setFilename] = useState("config.ef");
  const [logs, setLogs] = useState<string[]>([]);
  const [parsed, setParsed] = useState<any>(null);
  const [pipelineResults, setPipelineResults] = useState<any>(null);
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
                {/* AST Viewer */}
                {pipelineResults.ast && <ASTViewer ast={pipelineResults.ast} />}

                {/* IR Graph Viewer */}
                {pipelineResults.ir_graph && (
                  <IRGraphViewer irGraph={pipelineResults.ir_graph} />
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
      </main>
    </div>
  );
}
