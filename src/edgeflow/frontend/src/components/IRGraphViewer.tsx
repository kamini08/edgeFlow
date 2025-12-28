import React from "react";

interface IRNode {
  id: string;
  node_type: string;
  dependencies: string[];
  metadata?: Record<string, any>;
}

interface IREdge {
  from: string;
  to: string;
  data_type?: string;
}

interface IRGraph {
  nodes: IRNode[];
  edges: IREdge[];
}

interface IRGraphViewerProps {
  irGraph: IRGraph;
}

const IRGraphViewer: React.FC<IRGraphViewerProps> = ({ irGraph }) => {
  const getNodeColor = (nodeType: string): string => {
    const colors: Record<string, string> = {
      input: "bg-blue-100 border-blue-300",
      model: "bg-green-100 border-green-300",
      quantize: "bg-purple-100 border-purple-300",
      fusion: "bg-orange-100 border-orange-300",
      schedule: "bg-red-100 border-red-300",
      output: "bg-gray-100 border-gray-300",
      preprocess: "bg-yellow-100 border-yellow-300",
      postprocess: "bg-indigo-100 border-indigo-300",
    };
    return colors[nodeType] || "bg-gray-100 border-gray-300";
  };

  const getNodeIcon = (nodeType: string): string => {
    const icons: Record<string, string> = {
      input: "📥",
      model: "🧠",
      quantize: "⚡",
      fusion: "🔗",
      schedule: "⏱️",
      output: "📤",
      preprocess: "🔄",
      postprocess: "✨",
    };
    return icons[nodeType] || "📦";
  };

  return (
    <div className="bg-gray-50 border rounded-lg p-4">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Intermediate Representation Graph
      </h3>

      {/* Graph Overview */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-white border rounded p-3">
          <h4 className="font-semibold text-gray-900 mb-2">
            Nodes ({irGraph.nodes.length})
          </h4>
          <div className="space-y-2">
            {irGraph.nodes.map((node) => (
              <div
                key={node.id}
                className={`flex items-center gap-2 p-2 border rounded ${getNodeColor(
                  node.node_type
                )}`}>
                <span className="text-lg">{getNodeIcon(node.node_type)}</span>
                <div>
                  <div className="font-mono text-sm font-semibold">
                    {node.id}
                  </div>
                  <div className="text-xs text-gray-600">{node.node_type}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white border rounded p-3">
          <h4 className="font-semibold text-gray-900 mb-2">
            Edges ({irGraph.edges.length})
          </h4>
          <div className="space-y-1">
            {irGraph.edges.map((edge, index) => (
              <div key={index} className="flex items-center gap-2 text-sm">
                <span className="font-mono">{edge.from}</span>
                <span className="text-gray-400">→</span>
                <span className="font-mono">{edge.to}</span>
                {edge.data_type && (
                  <span className="text-xs text-gray-500 bg-gray-100 px-1 rounded">
                    {edge.data_type}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Node Details */}
      <div className="bg-white border rounded p-4">
        <h4 className="font-semibold text-gray-900 mb-3">Node Details</h4>
        <div className="space-y-3">
          {irGraph.nodes.map((node) => (
            <div key={node.id} className="border-l-4 border-blue-500 pl-4">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-lg">{getNodeIcon(node.node_type)}</span>
                <span className="font-mono font-semibold">{node.id}</span>
                <span className="text-sm text-gray-500">
                  ({node.node_type})
                </span>
              </div>
              {node.dependencies.length > 0 && (
                <div className="text-sm text-gray-600">
                  <span className="font-medium">Dependencies:</span>{" "}
                  {node.dependencies.join(", ")}
                </div>
              )}
              {node.metadata && Object.keys(node.metadata).length > 0 && (
                <div className="text-sm text-gray-600">
                  <span className="font-medium">Metadata:</span>
                  <pre className="mt-1 text-xs bg-gray-50 p-2 rounded">
                    {JSON.stringify(node.metadata, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default IRGraphViewer;
