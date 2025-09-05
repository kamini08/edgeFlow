import React from 'react';
import { render } from '@testing-library/react';
import OptimizationResults from '../../components/OptimizationResults';

describe('OptimizationResults', () => {
  test('renders metrics', () => {
    const { getByText } = render(
      <OptimizationResults
        originalStats={{ size_mb: 10, latency_ms: 100 }}
        optimizedStats={{ size_mb: 5, latency_ms: 70 }}
        improvement={{ size_reduction: 0.5, speedup: 100/70 }}
      />
    );
    expect(getByText(/Optimization Results/i)).toBeInTheDocument();
    expect(getByText(/Original Size: 10/)).toBeInTheDocument();
    expect(getByText(/Optimized Size: 5/)).toBeInTheDocument();
  });
});

