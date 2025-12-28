import React, { useState } from "react";

interface ValidationIssue {
  level: string;
  category: string;
  severity: string;
  message: string;
  details?: string;
  suggestions?: string[];
}

interface ValidationReport {
  package_path: string;
  device_type: string;
  validation_level: string;
  overall_result: string;
  issues: ValidationIssue[];
  metrics: Record<string, any>;
  recommendations: string[];
}

interface ValidationResultsViewerProps {
  validationReport: ValidationReport;
}

const ValidationResultsViewer: React.FC<ValidationResultsViewerProps> = ({
  validationReport,
}) => {
  const [selectedIssue, setSelectedIssue] = useState<ValidationIssue | null>(
    null
  );
  const [filterSeverity, setFilterSeverity] = useState<string>("all");

  const getSeverityIcon = (severity: string): string => {
    const icons: Record<string, string> = {
      fail: "❌",
      warn: "⚠️",
      pass: "✅",
      skip: "⏭️",
    };
    return icons[severity] || "❓";
  };

  const getSeverityColor = (severity: string): string => {
    const colors: Record<string, string> = {
      fail: "bg-red-50 border-red-200 text-red-800",
      warn: "bg-yellow-50 border-yellow-200 text-yellow-800",
      pass: "bg-green-50 border-green-200 text-green-800",
      skip: "bg-gray-50 border-gray-200 text-gray-800",
    };
    return colors[severity] || "bg-gray-50 border-gray-200 text-gray-800";
  };

  const getOverallResultColor = (result: string): string => {
    const colors: Record<string, string> = {
      pass: "bg-green-100 text-green-800",
      warn: "bg-yellow-100 text-yellow-800",
      fail: "bg-red-100 text-red-800",
    };
    return colors[result] || "bg-gray-100 text-gray-800";
  };

  const filteredIssues =
    filterSeverity === "all"
      ? validationReport.issues
      : validationReport.issues.filter(
          (issue) => issue.severity === filterSeverity
        );

  const issueStats = validationReport.issues.reduce((acc, issue) => {
    acc[issue.severity] = (acc[issue.severity] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="bg-gray-50 border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Deployment Validation Results
        </h3>
        <div
          className={`px-3 py-1 rounded-full text-sm font-medium ${getOverallResultColor(
            validationReport.overall_result
          )}`}>
          {getSeverityIcon(validationReport.overall_result)}{" "}
          {validationReport.overall_result.toUpperCase()}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-4">
          {/* Validation Summary */}
          <div className="bg-white border rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">
              Validation Summary
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">
                  {validationReport.issues.length}
                </div>
                <div className="text-sm text-gray-600">Total Issues</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {issueStats.fail || 0}
                </div>
                <div className="text-sm text-gray-600">Critical</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-yellow-600">
                  {issueStats.warn || 0}
                </div>
                <div className="text-sm text-gray-600">Warnings</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {issueStats.pass || 0}
                </div>
                <div className="text-sm text-gray-600">Passed</div>
              </div>
            </div>
          </div>

          {/* Issues List */}
          <div className="bg-white border rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-semibold text-gray-900">Issues</h4>
              <div className="flex gap-2">
                <button
                  className={`px-2 py-1 text-xs rounded ${
                    filterSeverity === "all"
                      ? "bg-blue-500 text-white"
                      : "bg-gray-200 text-gray-700"
                  }`}
                  onClick={() => setFilterSeverity("all")}>
                  All ({validationReport.issues.length})
                </button>
                <button
                  className={`px-2 py-1 text-xs rounded ${
                    filterSeverity === "fail"
                      ? "bg-red-500 text-white"
                      : "bg-gray-200 text-gray-700"
                  }`}
                  onClick={() => setFilterSeverity("fail")}>
                  Critical ({issueStats.fail || 0})
                </button>
                <button
                  className={`px-2 py-1 text-xs rounded ${
                    filterSeverity === "warn"
                      ? "bg-yellow-500 text-white"
                      : "bg-gray-200 text-gray-700"
                  }`}
                  onClick={() => setFilterSeverity("warn")}>
                  Warnings ({issueStats.warn || 0})
                </button>
              </div>
            </div>

            <div className="space-y-2">
              {filteredIssues.map((issue, index) => (
                <div
                  key={index}
                  className={`p-3 border rounded-lg cursor-pointer transition-all ${
                    selectedIssue === issue
                      ? "ring-2 ring-blue-500 shadow-lg"
                      : "hover:shadow-md"
                  } ${getSeverityColor(issue.severity)}`}
                  onClick={() => setSelectedIssue(issue)}>
                  <div className="flex items-start gap-3">
                    <span className="text-lg">
                      {getSeverityIcon(issue.severity)}
                    </span>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-semibold text-sm">
                          {issue.category}
                        </span>
                        <span className="text-xs opacity-75">
                          ({issue.level})
                        </span>
                      </div>
                      <div className="text-sm">{issue.message}</div>
                      {issue.details && (
                        <div className="text-xs opacity-75 mt-1">
                          {issue.details}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Recommendations */}
          {validationReport.recommendations.length > 0 && (
            <div className="bg-white border rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">
                Recommendations
              </h4>
              <div className="space-y-2">
                {validationReport.recommendations.map((rec, index) => (
                  <div key={index} className="flex items-start gap-2 text-sm">
                    <span className="text-blue-500 mt-0.5">💡</span>
                    <span>{rec}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Validation Info */}
          <div className="bg-white border rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">
              Validation Info
            </h4>
            <div className="space-y-2 text-sm">
              <div>
                <span className="font-medium text-gray-600">Package:</span>
                <div className="font-mono text-xs mt-1 break-all">
                  {validationReport.package_path}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-600">Device:</span>
                <span className="ml-2 font-mono">
                  {validationReport.device_type}
                </span>
              </div>
              <div>
                <span className="font-medium text-gray-600">Level:</span>
                <span className="ml-2 font-mono">
                  {validationReport.validation_level}
                </span>
              </div>
            </div>
          </div>

          {/* Metrics */}
          {Object.keys(validationReport.metrics).length > 0 && (
            <div className="bg-white border rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">Metrics</h4>
              <div className="space-y-2 text-sm">
                {Object.entries(validationReport.metrics).map(
                  ([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="font-medium text-gray-600">{key}:</span>
                      <span className="font-mono">
                        {typeof value === "number"
                          ? value.toFixed(2)
                          : String(value)}
                      </span>
                    </div>
                  )
                )}
              </div>
            </div>
          )}

          {/* Selected Issue Details */}
          {selectedIssue && (
            <div className="bg-white border rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">
                Issue Details
              </h4>
              <div className="space-y-2 text-sm">
                <div>
                  <span className="font-medium text-gray-600">Severity:</span>
                  <span className="ml-2">
                    {getSeverityIcon(selectedIssue.severity)}{" "}
                    {selectedIssue.severity}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-gray-600">Category:</span>
                  <span className="ml-2 font-mono">
                    {selectedIssue.category}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-gray-600">Level:</span>
                  <span className="ml-2 font-mono">{selectedIssue.level}</span>
                </div>
                {selectedIssue.details && (
                  <div>
                    <span className="font-medium text-gray-600">Details:</span>
                    <div className="mt-1 text-xs bg-gray-50 p-2 rounded">
                      {selectedIssue.details}
                    </div>
                  </div>
                )}
                {selectedIssue.suggestions &&
                  selectedIssue.suggestions.length > 0 && (
                    <div>
                      <span className="font-medium text-gray-600">
                        Suggestions:
                      </span>
                      <div className="mt-1 space-y-1">
                        {selectedIssue.suggestions.map((suggestion, idx) => (
                          <div
                            key={idx}
                            className="text-xs bg-blue-50 p-2 rounded">
                            💡 {suggestion}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="bg-white border rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">Legend</h4>
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <span>❌</span>
                <span>Critical Issue</span>
              </div>
              <div className="flex items-center gap-2">
                <span>⚠️</span>
                <span>Warning</span>
              </div>
              <div className="flex items-center gap-2">
                <span>✅</span>
                <span>Passed Check</span>
              </div>
              <div className="flex items-center gap-2">
                <span>⏭️</span>
                <span>Skipped Check</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ValidationResultsViewer;

