import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export interface BenchmarkData {
  name: string;
  size?: number;
  latency?: number;
}

export interface BenchmarkChartProps {
  data: BenchmarkData[];
  type: 'size' | 'latency' | 'both';
}

const BenchmarkChart: React.FC<BenchmarkChartProps> = ({ data, type }) => {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        {(type === 'size' || type === 'both') && <Bar dataKey="size" fill="#8884d8" name="Size (MB)" />}
        {(type === 'latency' || type === 'both') && <Bar dataKey="latency" fill="#82ca9d" name="Latency (ms)" />}
      </BarChart>
    </ResponsiveContainer>
  );
};

export default BenchmarkChart;

