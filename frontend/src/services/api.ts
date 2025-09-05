import axios from 'axios';

// Default to relative path so Next.js rewrites can proxy to backend in Docker Compose
const api = axios.create({ baseURL: process.env.NEXT_PUBLIC_API_URL || '' });

export const compileConfig = (config_file: string, filename: string) =>
  api.post('/api/compile', { config_file, filename }).then(r => r.data);

export const compileConfigVerbose = (config_file: string, filename: string) =>
  api.post('/api/compile/verbose', { config_file, filename }).then(r => r.data);

export const optimizeModel = (model_file: string, config: Record<string, unknown>) =>
  api.post('/api/optimize', { model_file, config }).then(r => r.data);

export const benchmarkModels = (original_model: string, optimized_model: string) =>
  api.post('/api/benchmark', { original_model, optimized_model }).then(r => r.data);

export const getVersion = () => api.get('/api/version').then(r => r.data);
export const getHelp = () => api.get('/api/help').then(r => r.data);
export const getHealth = () => api.get('/api/health').then(r => r.data);
