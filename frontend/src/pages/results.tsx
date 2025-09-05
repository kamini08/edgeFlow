import React, { useState } from 'react';
import OptimizationResults, { ModelStats, ImprovementMetrics } from '../components/OptimizationResults';
import BenchmarkChart from '../components/BenchmarkChart';

export default function ResultsPage() {
  const [data] = useState({
    original: { size_mb: 10, latency_ms: 100 } as ModelStats,
    optimized: { size_mb: 5, latency_ms: 70 } as ModelStats,
    improvement: { size_reduction: 0.5, speedup: 100/70 } as ImprovementMetrics,
  });

  const chartData = [
    { name: 'Original', size: data.original.size_mb, latency: data.original.latency_ms },
    { name: 'Optimized', size: data.optimized.size_mb, latency: data.optimized.latency_ms },
  ];

  return (
    <main style={{ padding: 24 }}>
      <h2>Results</h2>
      <OptimizationResults originalStats={data.original} optimizedStats={data.optimized} improvement={data.improvement} />
      <BenchmarkChart data={chartData} type="both" />
    </main>
  );
}

