import React, { useState } from 'react';
import Link from 'next/link';
import OptimizationResults, { ModelStats, ImprovementMetrics } from '../components/OptimizationResults';
import BenchmarkChart from '../components/BenchmarkChart';

export default function ResultsPage() {
  const [data] = useState({
    original: { size_mb: 10, latency_ms: 100 } as ModelStats,
    optimized: { size_mb: 5, latency_ms: 70 } as ModelStats,
    improvement: { size_reduction: 0.5, speedup: 100 / 70 } as ImprovementMetrics,
  });

  const chartData = [
    { name: 'Original', size: data.original.size_mb, latency: data.original.latency_ms },
    { name: 'Optimized', size: data.optimized.size_mb, latency: data.optimized.latency_ms },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b bg-white">
        <div className="container-narrow flex items-center justify-between py-4">
          <h1 className="text-xl font-semibold text-gray-900">EdgeFlow</h1>
          <nav className="space-x-4 text-sm font-medium">
            <Link className="text-gray-700 hover:text-blue-600" href="/">Home</Link>
            <Link className="text-gray-700 hover:text-blue-600" href="/compile">Compile</Link>
          </nav>
        </div>
      </header>
      <main className="container-narrow space-y-6 py-10">
        <section className="card">
          <h2 className="mb-2 text-lg font-semibold text-gray-900">Results</h2>
          <OptimizationResults
            originalStats={data.original}
            optimizedStats={data.optimized}
            improvement={data.improvement}
          />
        </section>
        <section className="card">
          <h3 className="mb-2 text-base font-semibold text-gray-900">Benchmark</h3>
          <BenchmarkChart data={chartData} type="both" />
        </section>
      </main>
    </div>
  );
}
