import React from 'react';
import { render } from '@testing-library/react';
import OptimizationResults from '../../components/OptimizationResults';

describe('OptimizationResults', () => {
  test('renders metrics', () => {
    const { getByText } = render(
      <OptimizationResults
        originalStats={{ size_mb: 10, latency_ms: 100 }}
        optimizedStats={{ size_mb: 5, latency_ms: 70 }}
        improvement={{ size_reduction: 0.5, speedup: 100 / 70 }}
      />
    );
    // Headings
    expect(getByText(/Original/i)).toBeInTheDocument();
    expect(getByText(/Optimized/i)).toBeInTheDocument();
    expect(getByText(/Improvement/i)).toBeInTheDocument();
    // Values
    expect(getByText(/Size:\s*10\s*MB/i)).toBeInTheDocument();
    expect(getByText(/Latency:\s*100\s*ms/i)).toBeInTheDocument();
    expect(getByText(/Size:\s*5\s*MB/i)).toBeInTheDocument();
    expect(getByText(/Latency:\s*70\s*ms/i)).toBeInTheDocument();
    expect(getByText(/50\.00%/)).toBeInTheDocument();
    expect(getByText(/1\.43x/)).toBeInTheDocument();
  });
});
