import React, { useState } from 'react';
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
    <main style={{ padding: 24 }}>
      <h2>Compile Config</h2>
      <FileUpload onFileSelect={onFileSelect} acceptedFormats={[".ef"]} maxSize={5} />
      <ConfigEditor initialContent={content} onChange={setContent} />
      <button onClick={() => onCompile(false)}>Compile</button>
      <button onClick={() => onCompile(true)} style={{ marginLeft: 8 }}>Compile Verbose</button>
      {parsed && (
        <pre>{JSON.stringify(parsed, null, 2)}</pre>
      )}
      {logs.length > 0 && (
        <details><summary>Logs</summary><pre>{logs.join('\n')}</pre></details>
      )}
    </main>
  );
}

