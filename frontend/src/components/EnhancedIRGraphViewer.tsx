import React, { useState, useMemo } from "react";

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
  execution_order?: string[];
}

interface EnhancedIRGraphViewerProps {
  irGraph: IRGraph;
}

const EnhancedIRGraphViewer: React.FC<EnhancedIRGraphViewerProps> = ({
  irGraph,
}) => {
  const [selectedNode, setSelectedNode] = useState<IRNode | null>(null);
  const [viewMode, setViewMode] = useState<"graph" | "list" | "execution">(
    "graph"
  );
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  const getNodeColor = (nodeType: string): string => {
    const colors: Record<string, string> = {
      INPUT: "bg-blue-100 border-blue-300 text-blue-800",
      MODEL: "bg-green-100 border-green-300 text-green-800",
      QUANTIZE: "bg-purple-100 border-purple-300 text-purple-800",
      FUSION: "bg-orange-100 border-orange-300 text-orange-800",
      SCHEDULE: "bg-red-100 border-red-300 text-red-800",
      OUTPUT: "bg-gray-100 border-gray-300 text-gray-800",
      PREPROCESS: "bg-yellow-100 border-yellow-300 text-yellow-800",
      POSTPROCESS: "bg-indigo-100 border-indigo-300 text-indigo-800",
    };
    return colors[nodeType] || "bg-gray-100 border-gray-300 text-gray-800";
  };

  const getNodeIcon = (nodeType: string): string => {
    const icons: Record<string, string> = {
      INPUT: "📥",
      MODEL: "🧠",
      QUANTIZE: "⚡",
      FUSION: "🔗",
      SCHEDULE: "⏱️",
      OUTPUT: "📤",
      PREPROCESS: "🔄",
      POSTPROCESS: "✨",
    };
    return icons[nodeType] || "📦";
  };

  const getNodeTypeStats = useMemo(() => {
    const stats: Record<string, number> = {};
    irGraph.nodes.forEach((node) => {
      stats[node.node_type] = (stats[node.node_type] || 0) + 1;
    });
    return stats;
  }, [irGraph.nodes]);

  const getExecutionOrder = useMemo(() => {
    if (irGraph.execution_order) {
      return irGraph.execution_order;
    }

    // Simple topological sort if execution_order not provided
    const visited = new Set<string>();
    const result: string[] = [];

    const visit = (nodeId: string) => {
      if (visited.has(nodeId)) return;
      visited.add(nodeId);

      const node = irGraph.nodes.find((n) => n.id === nodeId);
      if (node) {
        node.dependencies.forEach((dep) => visit(dep));
        result.push(nodeId);
      }
    };

    irGraph.nodes.forEach((node) => visit(node.id));
    return result;
  }, [irGraph]);

  const renderGraphView = () => {
    return (
      <div className="bg-white border rounded-lg p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {irGraph.nodes.map((node) => (
            <div
              key={node.id}
              className={`p-3 border rounded-lg cursor-pointer transition-all ${
                selectedNode?.id === node.id
                  ? "ring-2 ring-blue-500 shadow-lg"
                  : "hover:shadow-md"
              } ${getNodeColor(node.node_type)}`}
              onClick={() => setSelectedNode(node)}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-lg">{getNodeIcon(node.node_type)}</span>
                <div>
                  <div className="font-mono font-semibold text-sm">
                    {node.id}
                  </div>
                  <div className="text-xs opacity-75">{node.node_type}</div>
                </div>
              </div>

              {node.dependencies.length > 0 && (
                <div className="text-xs">
                  <span className="font-medium">Dependencies:</span>
                  <div className="mt-1 space-y-1">
                    {node.dependencies.map((dep, idx) => (
                      <div
                        key={idx}
                        className="bg-white bg-opacity-50 px-2 py-1 rounded text-xs">
                        {dep}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Edge Visualization */}
        <div className="mt-6">
          <h4 className="font-semibold text-gray-900 mb-3">Data Flow</h4>
          <div className="space-y-2">
            {irGraph.edges.map((edge, index) => (
              <div
                key={index}
                className="flex items-center gap-2 p-2 bg-gray-50 rounded">
                <span className="font-mono text-sm">{edge.from}</span>
                <span className="text-gray-400">→</span>
                <span className="font-mono text-sm">{edge.to}</span>
                {edge.data_type && (
                  <span className="text-xs text-gray-500 bg-gray-200 px-2 py-1 rounded">
                    {edge.data_type}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderListView = () => {
    return (
      <div className="bg-white border rounded-lg p-4">
        <div className="space-y-3">
          {irGraph.nodes.map((node) => (
            <div
              key={node.id}
              className={`p-3 border rounded-lg cursor-pointer transition-all ${
                selectedNode?.id === node.id
                  ? "ring-2 ring-blue-500 shadow-lg"
                  : "hover:shadow-md"
              } ${getNodeColor(node.node_type)}`}
              onClick={() => setSelectedNode(node)}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-lg">{getNodeIcon(node.node_type)}</span>
                  <div>
                    <div className="font-mono font-semibold">{node.id}</div>
                    <div className="text-sm opacity-75">{node.node_type}</div>
                  </div>
                </div>
                <div className="text-right text-sm">
                  <div className="text-gray-600">
                    {node.dependencies.length} dependency
                    {node.dependencies.length !== 1 ? "ies" : "y"}
                  </div>
                </div>
              </div>

              {node.dependencies.length > 0 && (
                <div className="mt-2 text-sm">
                  <span className="font-medium">Dependencies:</span>
                  <span className="ml-2 font-mono">
                    {node.dependencies.join(", ")}
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderExecutionView = () => {
    return (
      <div className="bg-white border rounded-lg p-4">
        <div className="space-y-3">
          {getExecutionOrder.map((nodeId, index) => {
            const node = irGraph.nodes.find((n) => n.id === nodeId);
            if (!node) return null;

            return (
              <div
                key={nodeId}
                className={`p-3 border rounded-lg cursor-pointer transition-all ${
                  selectedNode?.id === node.id
                    ? "ring-2 ring-blue-500 shadow-lg"
                    : "hover:shadow-md"
                } ${getNodeColor(node.node_type)}`}
                onClick={() => setSelectedNode(node)}>
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-8 h-8 bg-blue-500 text-white rounded-full text-sm font-bold">
                    {index + 1}
                  </div>
                  <span className="text-lg">{getNodeIcon(node.node_type)}</span>
                  <div>
                    <div className="font-mono font-semibold">{node.id}</div>
                    <div className="text-sm opacity-75">{node.node_type}</div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-gray-50 border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Intermediate Representation Graph
        </h3>
        <div className="flex gap-2">
          <button
            className={`px-3 py-1 text-sm rounded ${
              viewMode === "graph"
                ? "bg-blue-500 text-white"
                : "bg-gray-200 text-gray-700"
            }`}
            onClick={() => setViewMode("graph")}>
            Graph
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
          <button
            className={`px-3 py-1 text-sm rounded ${
              viewMode === "execution"
                ? "bg-blue-500 text-white"
                : "bg-gray-200 text-gray-700"
            }`}
            onClick={() => setViewMode("execution")}>
            Execution
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Main Content */}
        <div className="lg:col-span-3">
          {viewMode === "graph" && renderGraphView()}
          {viewMode === "list" && renderListView()}
          {viewMode === "execution" && renderExecutionView()}
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Graph Statistics */}
          <div className="bg-white border rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">
              Graph Statistics
            </h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Total Nodes:</span>
                <span className="font-mono">{irGraph.nodes.length}</span>
              </div>
              <div className="flex justify-between">
                <span>Total Edges:</span>
                <span className="font-mono">{irGraph.edges.length}</span>
              </div>
              <div className="flex justify-between">
                <span>Node Types:</span>
                <span className="font-mono">
                  {Object.keys(getNodeTypeStats).length}
                </span>
              </div>
            </div>
          </div>

          {/* Node Type Distribution */}
          <div className="bg-white border rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">Node Types</h4>
            <div className="space-y-2">
              {Object.entries(getNodeTypeStats)
                .sort(([, a], [, b]) => b - a)
                .map(([type, count]) => (
                  <div
                    key={type}
                    className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <span>{getNodeIcon(type)}</span>
                      <span className="font-mono">{type}</span>
                    </div>
                    <span className="bg-gray-100 px-2 py-1 rounded text-xs">
                      {count}
                    </span>
                  </div>
                ))}
            </div>
          </div>

          {/* Selected Node Details */}
          {selectedNode && (
            <div className="bg-white border rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">Node Details</h4>
              <div className="space-y-2 text-sm">
                <div>
                  <span className="font-medium text-gray-600">ID:</span>
                  <span className="ml-2 font-mono">{selectedNode.id}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-600">Type:</span>
                  <span className="ml-2 font-mono">
                    {selectedNode.node_type}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-gray-600">
                    Dependencies:
                  </span>
                  <span className="ml-2">
                    {selectedNode.dependencies.length}
                  </span>
                </div>
                {selectedNode.dependencies.length > 0 && (
                  <div>
                    <span className="font-medium text-gray-600">
                      Depends on:
                    </span>
                    <div className="mt-1 space-y-1">
                      {selectedNode.dependencies.map((dep, idx) => (
                        <div
                          key={idx}
                          className="bg-gray-50 px-2 py-1 rounded text-xs font-mono">
                          {dep}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {selectedNode.metadata &&
                  Object.keys(selectedNode.metadata).length > 0 && (
                    <div>
                      <span className="font-medium text-gray-600">
                        Metadata:
                      </span>
                      <pre className="mt-1 text-xs bg-gray-50 p-2 rounded overflow-x-auto">
                        {JSON.stringify(selectedNode.metadata, null, 2)}
                      </pre>
                    </div>
                  )}
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="bg-white border rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">Legend</h4>
            <div className="space-y-1 text-xs">
              {Object.entries({
                INPUT: "📥",
                MODEL: "🧠",
                QUANTIZE: "⚡",
                FUSION: "🔗",
                SCHEDULE: "⏱️",
                OUTPUT: "📤",
                PREPROCESS: "🔄",
                POSTPROCESS: "✨",
              }).map(([type, icon]) => (
                <div key={type} className="flex items-center gap-2">
                  <span>{icon}</span>
                  <span className="font-mono">{type}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedIRGraphViewer;

