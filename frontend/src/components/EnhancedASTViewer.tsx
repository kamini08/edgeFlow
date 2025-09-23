import React, { useState } from "react";

interface ASTNode {
  type: string;
  value?: any;
  children?: ASTNode[];
  metadata?: Record<string, any>;
}

interface EnhancedASTViewerProps {
  ast: ASTNode;
}

const EnhancedASTViewer: React.FC<EnhancedASTViewerProps> = ({ ast }) => {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(
    new Set(["root"])
  );
  const [selectedNode, setSelectedNode] = useState<ASTNode | null>(null);

  const toggleNode = (nodeId: string) => {
    const newExpanded = new Set(expandedNodes);
    if (newExpanded.has(nodeId)) {
      newExpanded.delete(nodeId);
    } else {
      newExpanded.add(nodeId);
    }
    setExpandedNodes(newExpanded);
  };

  const getNodeColor = (nodeType: string): string => {
    const colors: Record<string, string> = {
      Program: "bg-blue-100 border-blue-300 text-blue-800",
      ModelStatement: "bg-green-100 border-green-300 text-green-800",
      QuantizeStatement: "bg-purple-100 border-purple-300 text-purple-800",
      TargetDeviceStatement: "bg-orange-100 border-orange-300 text-orange-800",
      OptimizeForStatement: "bg-yellow-100 border-yellow-300 text-yellow-800",
      BufferSizeStatement: "bg-indigo-100 border-indigo-300 text-indigo-800",
      MemoryLimitStatement: "bg-red-100 border-red-300 text-red-800",
      ConditionalStatement: "bg-pink-100 border-pink-300 text-pink-800",
      PipelineStatement: "bg-teal-100 border-teal-300 text-teal-800",
      FusionStatement: "bg-cyan-100 border-cyan-300 text-cyan-800",
      PruningStatement: "bg-emerald-100 border-emerald-300 text-emerald-800",
      PruningSparsityStatement:
        "bg-emerald-100 border-emerald-300 text-emerald-800",
      DeployPathStatement: "bg-gray-100 border-gray-300 text-gray-800",
    };
    return colors[nodeType] || "bg-gray-100 border-gray-300 text-gray-800";
  };

  const getNodeIcon = (nodeType: string): string => {
    const icons: Record<string, string> = {
      Program: "📋",
      ModelStatement: "🧠",
      QuantizeStatement: "⚡",
      TargetDeviceStatement: "📱",
      OptimizeForStatement: "🎯",
      BufferSizeStatement: "📦",
      MemoryLimitStatement: "💾",
      ConditionalStatement: "🔀",
      PipelineStatement: "🔄",
      FusionStatement: "🔗",
      PruningStatement: "✂️",
      PruningSparsityStatement: "📊",
      DeployPathStatement: "📁",
    };
    return icons[nodeType] || "📄";
  };

  const renderNode = (
    node: ASTNode,
    depth = 0,
    nodeId = "root"
  ): React.ReactNode => {
    const hasChildren = node.children && node.children.length > 0;
    const isExpanded = expandedNodes.has(nodeId);
    const isSelected = selectedNode === node;

    return (
      <div key={nodeId} className="select-none">
        <div
          className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
            isSelected
              ? "bg-blue-50 border-2 border-blue-300"
              : "hover:bg-gray-50"
          } ${getNodeColor(node.type)}`}
          onClick={() => {
            setSelectedNode(node);
            if (hasChildren) {
              toggleNode(nodeId);
            }
          }}>
          {hasChildren && (
            <button
              className="w-4 h-4 flex items-center justify-center text-xs font-bold"
              onClick={(e) => {
                e.stopPropagation();
                toggleNode(nodeId);
              }}>
              {isExpanded ? "−" : "+"}
            </button>
          )}
          {!hasChildren && <div className="w-4" />}

          <span className="text-lg">{getNodeIcon(node.type)}</span>

          <div className="flex-1">
            <span className="font-semibold text-sm">{node.type}</span>
            {node.value !== undefined && (
              <span className="ml-2 text-sm opacity-75">
                ={" "}
                {typeof node.value === "string"
                  ? `"${node.value}"`
                  : String(node.value)}
              </span>
            )}
          </div>

          {hasChildren && (
            <span className="text-xs opacity-60">
              {node.children!.length} child
              {node.children!.length !== 1 ? "ren" : ""}
            </span>
          )}
        </div>

        {hasChildren && isExpanded && (
          <div className="ml-6 mt-1 space-y-1">
            {node.children!.map((child, index) =>
              renderNode(child, depth + 1, `${nodeId}-${index}`)
            )}
          </div>
        )}
      </div>
    );
  };

  const getASTStats = (
    node: ASTNode
  ): { totalNodes: number; nodeTypes: Record<string, number> } => {
    let totalNodes = 1;
    const nodeTypes: Record<string, number> = { [node.type]: 1 };

    if (node.children) {
      for (const child of node.children) {
        const childStats = getASTStats(child);
        totalNodes += childStats.totalNodes;
        for (const [type, count] of Object.entries(childStats.nodeTypes)) {
          nodeTypes[type] = (nodeTypes[type] || 0) + count;
        }
      }
    }

    return { totalNodes, nodeTypes };
  };

  const astStats = getASTStats(ast);

  return (
    <div className="bg-gray-50 border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Abstract Syntax Tree
        </h3>
        <div className="flex gap-4 text-sm text-gray-600">
          <span>Total Nodes: {astStats.totalNodes}</span>
          <span>Node Types: {Object.keys(astStats.nodeTypes).length}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Tree Visualization */}
        <div className="lg:col-span-2">
          <div className="bg-white border rounded-lg p-4 max-h-96 overflow-y-auto">
            <div className="text-green-600 mb-3 font-semibold">
              AST Structure
            </div>
            {renderNode(ast)}
          </div>
        </div>

        {/* Node Details & Statistics */}
        <div className="space-y-4">
          {/* Selected Node Details */}
          {selectedNode && (
            <div className="bg-white border rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">
                Selected Node
              </h4>
              <div className="space-y-2 text-sm">
                <div>
                  <span className="font-medium text-gray-600">Type:</span>
                  <span className="ml-2 font-mono">{selectedNode.type}</span>
                </div>
                {selectedNode.value !== undefined && (
                  <div>
                    <span className="font-medium text-gray-600">Value:</span>
                    <span className="ml-2 font-mono">
                      {typeof selectedNode.value === "string"
                        ? `"${selectedNode.value}"`
                        : String(selectedNode.value)}
                    </span>
                  </div>
                )}
                {selectedNode.children && (
                  <div>
                    <span className="font-medium text-gray-600">Children:</span>
                    <span className="ml-2">{selectedNode.children.length}</span>
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

          {/* Node Type Statistics */}
          <div className="bg-white border rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">
              Node Statistics
            </h4>
            <div className="space-y-2">
              {Object.entries(astStats.nodeTypes)
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

          {/* Legend */}
          <div className="bg-white border rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">Legend</h4>
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <span>+</span>
                <span>Expandable node</span>
              </div>
              <div className="flex items-center gap-2">
                <span>−</span>
                <span>Collapsible node</span>
              </div>
              <div className="flex items-center gap-2">
                <span>📄</span>
                <span>Leaf node</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedASTViewer;
