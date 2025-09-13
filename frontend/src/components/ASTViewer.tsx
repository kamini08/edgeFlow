import React from "react";

interface ASTNode {
  type: string;
  value: any;
  children?: ASTNode[];
}

interface ASTViewerProps {
  ast: ASTNode;
}

const ASTViewer: React.FC<ASTViewerProps> = ({ ast }) => {
  const renderNode = (node: ASTNode, depth = 0): React.ReactNode => {
    const indent = "  ".repeat(depth);
    const hasChildren = node.children && node.children.length > 0;

    return (
      <div key={`${node.type}-${depth}`} className="ml-4">
        <div className="flex items-center gap-2 py-1">
          <span className="text-blue-600 font-mono text-sm">
            {indent}
            {node.type}
          </span>
          {node.value && (
            <span className="text-gray-600 text-sm">
              ={" "}
              {typeof node.value === "string"
                ? `"${node.value}"`
                : String(node.value)}
            </span>
          )}
        </div>
        {hasChildren && (
          <div className="ml-4">
            {node.children!.map((child, index) => renderNode(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-gray-50 border rounded-lg p-4">
      <h3 className="text-lg font-semibold text-gray-900 mb-3">
        Abstract Syntax Tree
      </h3>
      <div className="bg-white border rounded p-3 font-mono text-sm">
        <div className="text-green-600 mb-2">AST Root</div>
        {renderNode(ast)}
      </div>
    </div>
  );
};

export default ASTViewer;
