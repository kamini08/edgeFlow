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
    <div>
      <h3>Optimization Results</h3>
      <p>Original Size: {originalStats.size_mb} MB, Latency: {originalStats.latency_ms} ms</p>
      <p>Optimized Size: {optimizedStats.size_mb} MB, Latency: {optimizedStats.latency_ms} ms</p>
      <p>Size Reduction: {(improvement.size_reduction * 100).toFixed(2)}%</p>
      <p>Speedup: {improvement.speedup.toFixed(2)}x</p>
    </div>
  );
};

export default OptimizationResults;

