import React, { useState } from "react";

interface GeneratedCodeViewerProps {
  generatedCode: Record<string, string>;
}

const GeneratedCodeViewer: React.FC<GeneratedCodeViewerProps> = ({
  generatedCode,
}) => {
  const [activeTab, setActiveTab] = useState<string>(
    Object.keys(generatedCode)[0] || ""
  );

  const getLanguageFromBackend = (backend: string): string => {
    const languages: Record<string, string> = {
      python: "python",
      cpp: "cpp",
      onnx: "python",
      tensorrt: "python",
    };
    return languages[backend] || "text";
  };

  const getBackendIcon = (backend: string): string => {
    const icons: Record<string, string> = {
      python: "🐍",
      cpp: "⚙️",
      onnx: "🔷",
      tensorrt: "🚀",
    };
    return icons[backend] || "📄";
  };

  const getBackendDescription = (backend: string): string => {
    const descriptions: Record<string, string> = {
      python: "Python inference code with TensorFlow/Keras",
      cpp: "C++ inference code for high performance",
      onnx: "ONNX Runtime optimized inference",
      tensorrt: "TensorRT accelerated inference for NVIDIA GPUs",
    };
    return descriptions[backend] || "Generated inference code";
  };

  return (
    <div className="bg-gray-50 border rounded-lg p-4">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Generated Code
      </h3>

      {/* Tab Navigation */}
      <div className="flex space-x-1 mb-4 bg-gray-100 rounded-lg p-1">
        {Object.keys(generatedCode).map((backend) => (
          <button
            key={backend}
            onClick={() => setActiveTab(backend)}
            className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === backend
                ? "bg-white text-gray-900 shadow-sm"
                : "text-gray-600 hover:text-gray-900"
            }`}>
            <span>{getBackendIcon(backend)}</span>
            <span className="capitalize">{backend}</span>
          </button>
        ))}
      </div>

      {/* Active Tab Content */}
      {activeTab && generatedCode[activeTab] && (
        <div className="bg-white border rounded-lg">
          {/* Tab Header */}
          <div className="border-b bg-gray-50 px-4 py-3 rounded-t-lg">
            <div className="flex items-center gap-2">
              <span className="text-lg">{getBackendIcon(activeTab)}</span>
              <div>
                <h4 className="font-semibold text-gray-900 capitalize">
                  {activeTab} Backend
                </h4>
                <p className="text-sm text-gray-600">
                  {getBackendDescription(activeTab)}
                </p>
              </div>
            </div>
          </div>

          {/* Code Content */}
          <div className="relative">
            <pre className="p-4 overflow-auto max-h-96 text-sm">
              <code className={`language-${getLanguageFromBackend(activeTab)}`}>
                {generatedCode[activeTab]}
              </code>
            </pre>

            {/* Copy Button */}
            <button
              onClick={() => {
                navigator.clipboard.writeText(generatedCode[activeTab]);
                // You could add a toast notification here
              }}
              className="absolute top-2 right-2 bg-gray-100 hover:bg-gray-200 text-gray-600 px-3 py-1 rounded text-xs font-medium transition-colors">
              Copy
            </button>
          </div>
        </div>
      )}

      {/* Backend Summary */}
      <div className="mt-4 grid grid-cols-2 gap-3">
        {Object.keys(generatedCode).map((backend) => (
          <div key={backend} className="bg-white border rounded p-3">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-lg">{getBackendIcon(backend)}</span>
              <span className="font-semibold capitalize">{backend}</span>
            </div>
            <p className="text-sm text-gray-600">
              {getBackendDescription(backend)}
            </p>
            <div className="mt-2 text-xs text-gray-500">
              {generatedCode[backend].length} characters
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default GeneratedCodeViewer;
