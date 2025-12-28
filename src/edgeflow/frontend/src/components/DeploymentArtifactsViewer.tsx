import React, { useState } from "react";

interface DeploymentArtifact {
  artifact_path: string;
  artifact_type: string;
  device_type: string;
  package_format: string;
  size_mb: number;
  dependencies: string[];
  metadata: Record<string, any>;
}

interface DeploymentArtifactsViewerProps {
  artifacts: DeploymentArtifact[];
  deviceType: string;
}

const DeploymentArtifactsViewer: React.FC<DeploymentArtifactsViewerProps> = ({
  artifacts,
  deviceType,
}) => {
  const [selectedArtifact, setSelectedArtifact] =
    useState<DeploymentArtifact | null>(null);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");

  const getArtifactIcon = (artifactType: string): string => {
    const icons: Record<string, string> = {
      model_binary: "🧠",
      inference_code: "💻",
      dependencies: "📦",
      deployment_scripts: "🚀",
      final_package: "📋",
    };
    return icons[artifactType] || "📄";
  };

  const getArtifactColor = (artifactType: string): string => {
    const colors: Record<string, string> = {
      model_binary: "bg-green-100 border-green-300 text-green-800",
      inference_code: "bg-blue-100 border-blue-300 text-blue-800",
      dependencies: "bg-purple-100 border-purple-300 text-purple-800",
      deployment_scripts: "bg-orange-100 border-orange-300 text-orange-800",
      final_package: "bg-gray-100 border-gray-300 text-gray-800",
    };
    return colors[artifactType] || "bg-gray-100 border-gray-300 text-gray-800";
  };

  const formatFileSize = (sizeMb: number): string => {
    if (sizeMb < 1) {
      return `${(sizeMb * 1024).toFixed(1)} KB`;
    }
    return `${sizeMb.toFixed(1)} MB`;
  };

  const totalSize = artifacts.reduce(
    (sum, artifact) => sum + artifact.size_mb,
    0
  );

  const renderGridView = () => {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {artifacts.map((artifact, index) => (
          <div
            key={index}
            className={`p-4 border rounded-lg cursor-pointer transition-all ${
              selectedArtifact === artifact
                ? "ring-2 ring-blue-500 shadow-lg"
                : "hover:shadow-md"
            } ${getArtifactColor(artifact.artifact_type)}`}
            onClick={() => setSelectedArtifact(artifact)}>
            <div className="flex items-center gap-3 mb-3">
              <span className="text-2xl">
                {getArtifactIcon(artifact.artifact_type)}
              </span>
              <div>
                <div className="font-semibold text-sm capitalize">
                  {artifact.artifact_type.replace("_", " ")}
                </div>
                <div className="text-xs opacity-75">
                  {artifact.package_format}
                </div>
              </div>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="font-medium">Size:</span>
                <span className="font-mono">
                  {formatFileSize(artifact.size_mb)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="font-medium">Dependencies:</span>
                <span>{artifact.dependencies.length}</span>
              </div>
            </div>

            <div className="mt-3 text-xs opacity-75 break-all">
              {artifact.artifact_path.split("/").pop()}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderListView = () => {
    return (
      <div className="space-y-3">
        {artifacts.map((artifact, index) => (
          <div
            key={index}
            className={`p-4 border rounded-lg cursor-pointer transition-all ${
              selectedArtifact === artifact
                ? "ring-2 ring-blue-500 shadow-lg"
                : "hover:shadow-md"
            } ${getArtifactColor(artifact.artifact_type)}`}
            onClick={() => setSelectedArtifact(artifact)}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-xl">
                  {getArtifactIcon(artifact.artifact_type)}
                </span>
                <div>
                  <div className="font-semibold capitalize">
                    {artifact.artifact_type.replace("_", " ")}
                  </div>
                  <div className="text-sm opacity-75">
                    {artifact.package_format}
                  </div>
                </div>
              </div>
              <div className="text-right text-sm">
                <div className="font-mono">
                  {formatFileSize(artifact.size_mb)}
                </div>
                <div className="text-xs opacity-75">
                  {artifact.dependencies.length} deps
                </div>
              </div>
            </div>

            <div className="mt-2 text-xs opacity-75 break-all">
              {artifact.artifact_path}
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="bg-gray-50 border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Deployment Artifacts
        </h3>
        <div className="flex gap-2">
          <button
            className={`px-3 py-1 text-sm rounded ${
              viewMode === "grid"
                ? "bg-blue-500 text-white"
                : "bg-gray-200 text-gray-700"
            }`}
            onClick={() => setViewMode("grid")}>
            Grid
          </button>
          <button
            className={`px-3 py-1 text-sm rounded ${
              viewMode === "list"
                ? "bg-blue-500 text-white"
                : "bg-gray-200 text-gray-700"
            }`}
            onClick={() => setViewMode("list")}>
            List
          </button>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-white border rounded-lg p-4 mb-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {artifacts.length}
            </div>
            <div className="text-sm text-gray-600">Total Artifacts</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {formatFileSize(totalSize)}
            </div>
            <div className="text-sm text-gray-600">Total Size</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {deviceType}
            </div>
            <div className="text-sm text-gray-600">Target Device</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {artifacts.reduce(
                (sum, artifact) => sum + artifact.dependencies.length,
                0
              )}
            </div>
            <div className="text-sm text-gray-600">Total Dependencies</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Main Content */}
        <div className="lg:col-span-2">
          <div className="bg-white border rounded-lg p-4">
            {viewMode === "grid" ? renderGridView() : renderListView()}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Selected Artifact Details */}
          {selectedArtifact && (
            <div className="bg-white border rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">
                Artifact Details
              </h4>
              <div className="space-y-2 text-sm">
                <div>
                  <span className="font-medium text-gray-600">Type:</span>
                  <span className="ml-2 font-mono capitalize">
                    {selectedArtifact.artifact_type.replace("_", " ")}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-gray-600">Size:</span>
                  <span className="ml-2 font-mono">
                    {formatFileSize(selectedArtifact.size_mb)}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-gray-600">Format:</span>
                  <span className="ml-2 font-mono">
                    {selectedArtifact.package_format}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-gray-600">Device:</span>
                  <span className="ml-2 font-mono">
                    {selectedArtifact.device_type}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-gray-600">Path:</span>
                  <div className="mt-1 text-xs bg-gray-50 p-2 rounded break-all">
                    {selectedArtifact.artifact_path}
                  </div>
                </div>
                {selectedArtifact.dependencies.length > 0 && (
                  <div>
                    <span className="font-medium text-gray-600">
                      Dependencies:
                    </span>
                    <div className="mt-1 space-y-1">
                      {selectedArtifact.dependencies.map((dep, idx) => (
                        <div
                          key={idx}
                          className="bg-gray-50 px-2 py-1 rounded text-xs font-mono">
                          {dep}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {selectedArtifact.metadata &&
                  Object.keys(selectedArtifact.metadata).length > 0 && (
                    <div>
                      <span className="font-medium text-gray-600">
                        Metadata:
                      </span>
                      <pre className="mt-1 text-xs bg-gray-50 p-2 rounded overflow-x-auto">
                        {JSON.stringify(selectedArtifact.metadata, null, 2)}
                      </pre>
                    </div>
                  )}
              </div>
            </div>
          )}

          {/* Artifact Types */}
          <div className="bg-white border rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">Artifact Types</h4>
            <div className="space-y-2">
              {Array.from(new Set(artifacts.map((a) => a.artifact_type))).map(
                (type) => {
                  const count = artifacts.filter(
                    (a) => a.artifact_type === type
                  ).length;
                  const totalSize = artifacts
                    .filter((a) => a.artifact_type === type)
                    .reduce((sum, a) => sum + a.size_mb, 0);

                  return (
                    <div
                      key={type}
                      className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <span>{getArtifactIcon(type)}</span>
                        <span className="capitalize">
                          {type.replace("_", " ")}
                        </span>
                      </div>
                      <div className="text-right">
                        <div className="font-mono text-xs">{count}</div>
                        <div className="font-mono text-xs">
                          {formatFileSize(totalSize)}
                        </div>
                      </div>
                    </div>
                  );
                }
              )}
            </div>
          </div>

          {/* Legend */}
          <div className="bg-white border rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">Legend</h4>
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <span>🧠</span>
                <span>Model Binary</span>
              </div>
              <div className="flex items-center gap-2">
                <span>💻</span>
                <span>Inference Code</span>
              </div>
              <div className="flex items-center gap-2">
                <span>📦</span>
                <span>Dependencies</span>
              </div>
              <div className="flex items-center gap-2">
                <span>🚀</span>
                <span>Deployment Scripts</span>
              </div>
              <div className="flex items-center gap-2">
                <span>📋</span>
                <span>Final Package</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeploymentArtifactsViewer;

