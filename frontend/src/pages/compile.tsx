import React, { useState } from 'react';
import Link from 'next/link';
import FileUpload from '../components/FileUpload';
import ConfigEditor from '../components/ConfigEditor';
import { compileConfig, compileConfigVerbose } from '../services/api';

export default function CompilePage() {
  const [content, setContent] = useState('');
  const [filename, setFilename] = useState('config.ef');
  const [logs, setLogs] = useState<string[]>([]);
  const [parsed, setParsed] = useState<any>(null);

  const onFileSelect = (file: File) => {
    setFilename(file.name);
    file.text().then(setContent);
  };

  const onCompile = async (verbose = false) => {
    const fn = verbose ? compileConfigVerbose : compileConfig;
    const res = await fn(content, filename);
    setParsed(res.parsed_config);
    setLogs(res.logs || []);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b bg-white">
        <div className="container-narrow flex items-center justify-between py-4">
          <h1 className="text-xl font-semibold text-gray-900">EdgeFlow</h1>
          <nav className="space-x-4 text-sm font-medium">
            <Link className="text-gray-700 hover:text-blue-600" href="/">Home</Link>
            <Link className="text-gray-700 hover:text-blue-600" href="/results">Results</Link>
          </nav>
        </div>
      </header>
      <main className="container-narrow py-10 space-y-6">
        <section className="card space-y-4">
          <h2 className="text-lg font-semibold text-gray-900">Compile Config</h2>
          <div>
            <FileUpload onFileSelect={onFileSelect} acceptedFormats={['.ef']} maxSize={5} />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-700">{filename}</label>
            <ConfigEditor initialContent={content} onChange={setContent} />
          </div>
          <div className="flex gap-3">
            <button className="btn" onClick={() => onCompile(false)}>Compile</button>
            <button className="btn bg-gray-700 hover:bg-gray-800" onClick={() => onCompile(true)}>Compile Verbose</button>
          </div>
        </section>
        {parsed && (
          <section className="card">
            <h3 className="mb-2 text-base font-semibold text-gray-900">Parsed Config</h3>
            <pre className="overflow-auto whitespace-pre-wrap text-sm text-gray-800">{JSON.stringify(parsed, null, 2)}</pre>
          </section>
        )}
        {logs.length > 0 && (
          <section className="card">
            <h3 className="mb-2 text-base font-semibold text-gray-900">Logs</h3>
            <pre className="overflow-auto whitespace-pre-wrap text-sm text-gray-800">{logs.join('\n')}</pre>
          </section>
        )}
      </main>
    </div>
  );
}
