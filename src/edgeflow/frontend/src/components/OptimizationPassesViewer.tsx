import React from "react";

interface OptimizationPass {
  name: string;
  description: string;
  nodes_added?: number;
}

interface OptimizationPassesViewerProps {
  optimizationPasses: OptimizationPass[];
}

const OptimizationPassesViewer: React.FC<OptimizationPassesViewerProps> = ({
  optimizationPasses,
}) => {
  const getPassIcon = (passName: string): string => {
    const icons: Record<string, string> = {
      QuantizationPass: "⚡",
      FusionPass: "🔗",
      SchedulingPass: "⏱️",
      PruningPass: "✂️",
      KnowledgeDistillationPass: "🎓",
    };
    return icons[passName] || "🔧";
  };

  const getPassColor = (passName: string): string => {
    const colors: Record<string, string> = {
      QuantizationPass: "border-purple-300 bg-purple-50",
      FusionPass: "border-orange-300 bg-orange-50",
      SchedulingPass: "border-red-300 bg-red-50",
      PruningPass: "border-green-300 bg-green-50",
      KnowledgeDistillationPass: "border-blue-300 bg-blue-50",
    };
    return colors[passName] || "border-gray-300 bg-gray-50";
  };

  const getPassDescription = (passName: string): string => {
    const descriptions: Record<string, string> = {
      QuantizationPass:
        "Reduces model precision to decrease memory usage and improve inference speed",
      FusionPass:
        "Combines multiple operations into single kernels to reduce memory bandwidth",
      SchedulingPass:
        "Optimizes execution order and resource allocation for better performance",
      PruningPass:
        "Removes unnecessary weights and connections to reduce model size",
      KnowledgeDistillationPass:
        "Creates smaller models by transferring knowledge from larger models",
    };
    return (
      descriptions[passName] ||
      "Applies optimization transformations to the model"
    );
  };

  return (
    <div className="bg-gray-50 border rounded-lg p-4">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Optimization Passes ({optimizationPasses.length})
      </h3>

      <div className="space-y-3">
        {optimizationPasses.map((pass, index) => (
          <div
            key={pass.name}
            className={`border rounded-lg p-4 ${getPassColor(pass.name)}`}>
            <div className="flex items-start gap-3">
              <div className="flex items-center gap-2">
                <span className="text-2xl">{getPassIcon(pass.name)}</span>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-semibold text-gray-900">
                    {pass.name}
                  </span>
                  <span className="text-sm text-gray-500">#{index + 1}</span>
                </div>
              </div>
            </div>

            <div className="mt-2">
              <p className="text-gray-700 mb-2">{pass.description}</p>
              <p className="text-sm text-gray-600">
                {getPassDescription(pass.name)}
              </p>

              {pass.nodes_added && (
                <div className="mt-2 flex items-center gap-2">
                  <span className="text-xs bg-white px-2 py-1 rounded border">
                    +{pass.nodes_added} nodes
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Summary */}
      <div className="mt-4 bg-white border rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">
          Optimization Summary
        </h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Total Passes:</span>
            <span className="ml-2 font-semibold">
              {optimizationPasses.length}
            </span>
          </div>
          <div>
            <span className="text-gray-600">Nodes Added:</span>
            <span className="ml-2 font-semibold">
              {optimizationPasses.reduce(
                (sum, pass) => sum + (pass.nodes_added || 0),
                0
              )}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OptimizationPassesViewer;
