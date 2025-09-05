import { useState } from 'react';
import { compileConfig, optimizeModel, benchmarkModels } from '../services/api';

export function useEdgeFlow() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const compile = async (content: string, filename: string) => {
    setLoading(true); setError(null);
    try { return await compileConfig(content, filename); } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  };
  const optimize = async (modelB64: string, config: Record<string, unknown>) => {
    setLoading(true); setError(null);
    try { return await optimizeModel(modelB64, config); } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  };
  const benchmark = async (orig: string, opt: string) => {
    setLoading(true); setError(null);
    try { return await benchmarkModels(orig, opt); } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  };

  return { compile, optimize, benchmark, loading, error };
}

export default useEdgeFlow;

