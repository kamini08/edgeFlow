import React from 'react';

export interface ModelStats { size_mb: number; latency_ms: number }
export interface ImprovementMetrics { size_reduction: number; speedup: number }

export interface OptimizationResultsProps {
  originalStats: ModelStats;
  optimizedStats: ModelStats;
  improvement: ImprovementMetrics;
}

const OptimizationResults: React.FC<OptimizationResultsProps> = ({ originalStats, optimizedStats, improvement }) => {
  return (
    <div className="grid gap-3 text-sm text-gray-800 md:grid-cols-2">
      <div className="card">
        <h4 className="mb-2 font-semibold text-gray-900">Original</h4>
        <p>Size: {originalStats.size_mb} MB</p>
        <p>Latency: {originalStats.latency_ms} ms</p>
      </div>
      <div className="card">
        <h4 className="mb-2 font-semibold text-gray-900">Optimized</h4>
        <p>Size: {optimizedStats.size_mb} MB</p>
        <p>Latency: {optimizedStats.latency_ms} ms</p>
      </div>
      <div className="card md:col-span-2">
        <h4 className="mb-2 font-semibold text-gray-900">Improvement</h4>
        <div className="flex flex-wrap gap-6">
          <div>
            <div className="text-xs uppercase text-gray-500">Size Reduction</div>
            <div className="text-lg font-semibold">{(improvement.size_reduction * 100).toFixed(2)}%</div>
          </div>
          <div>
            <div className="text-xs uppercase text-gray-500">Speedup</div>
            <div className="text-lg font-semibold">{improvement.speedup.toFixed(2)}x</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OptimizationResults;
