import React, { useState } from 'react';
import { checkCompatibility } from '../services/api';
import { FiCheckCircle, FiAlertTriangle, FiX } from 'react-icons/fi';

interface CompatibilityCheckProps {
  modelFile: File | null;
  config: any;
  onCheckComplete: (result: any) => void;
}

export const CompatibilityCheck: React.FC<CompatibilityCheckProps> = ({
  modelFile,
  config,
  onCheckComplete,
}) => {
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleCheck = async () => {
    if (!modelFile && !config?.model_path && !config?.model) {
      alert('Please select a model file or set model_path in config');
      return;
    }

    setChecking(true);
    try {
      const response = await checkCompatibility(modelFile, config);
      setResult(response);
      onCheckComplete(response);
    } catch (error) {
      console.error('Compatibility check failed:', error);
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="compatibility-check p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-4">Device Compatibility Check</h2>

      <button
        onClick={handleCheck}
        disabled={checking || (!modelFile && !config?.model_path && !config?.model)}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
      >
        {checking ? 'Checking...' : 'Check Compatibility'}
      </button>

      {result && (
        <div className="mt-6">
          <div
            className={`flex items-center gap-2 text-lg font-semibold ${
              result.compatible ? 'text-green-600' : 'text-red-600'
            }`}
          >
            {result.compatible ? (
              <FiCheckCircle className="text-2xl" />
            ) : (
              <FiX className="text-2xl" />
            )}
            {result.compatible ? 'Compatible' : 'Not Compatible'}
          </div>

          <div className="mt-4">
            <div className="text-sm text-gray-600">Fit Score</div>
            <div className="w-full bg-gray-200 rounded-full h-4 mt-1">
              <div
                className={`h-4 rounded-full ${
                  result.fit_score > 70
                    ? 'bg-green-500'
                    : result.fit_score > 40
                    ? 'bg-yellow-500'
                    : 'bg-red-500'
                }`}
                style={{ width: `${result.fit_score}%` }}
              />
            </div>
            <div className="text-right text-sm mt-1">{Number(result.fit_score).toFixed(1)}%</div>
          </div>

          {result.issues?.length > 0 && (
            <div className="mt-4">
              <h3 className="font-semibold flex items-center gap-2">
                <FiAlertTriangle className="text-yellow-500" />
                Issues Found
              </h3>
              <ul className="list-disc list-inside mt-2 text-sm text-gray-700">
                {result.issues.map((issue: string, i: number) => (
                  <li key={i}>{issue}</li>
                ))}
              </ul>
            </div>
          )}

          {result.recommendations?.length > 0 && (
            <div className="mt-4">
              <h3 className="font-semibold">💡 Recommendations</h3>
              <ul className="list-disc list-inside mt-2 text-sm text-gray-700">
                {result.recommendations.map((rec: string, i: number) => (
                  <li key={i}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CompatibilityCheck;

