import axios from 'axios';

// Mock axios and its create() instance method used by api.ts
jest.mock('axios', () => {
  const post = jest.fn();
  const create = jest.fn(() => ({ post }));
  return { __esModule: true, default: { create }, create };
});

// Import after mocks so api.ts picks up the mocked instance
import { compileConfig } from './api';

describe('Parser API Integration', () => {
  const instance = (axios as any).create();
  beforeEach(() => {
    instance.post.mockReset();
  });

  it('should parse valid configuration', async () => {
    const config = `\n  model_path = "model.tflite"\n  quantize = int8\n`;

    instance.post.mockResolvedValueOnce({
      data: {
        success: true,
        parsed_config: { model_path: 'model.tflite', quantize: 'int8' },
      },
    } as any);

    const result = await compileConfig(config, 'test.ef');
    expect(result.success).toBe(true);
    expect(result.parsed_config.model_path).toBe('model.tflite');
    expect(instance.post).toHaveBeenCalledWith('/api/compile', {
      config_file: config,
      filename: 'test.ef',
    });
  });

  it('should handle parsing errors', async () => {
    const invalid = 'invalid syntax ===';
    instance.post.mockRejectedValueOnce(new Error('Bad Request'));

    await expect(compileConfig(invalid, 'test.ef')).rejects.toThrow();
  });
});
