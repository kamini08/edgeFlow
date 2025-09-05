import React from 'react';

export interface ConfigEditorProps {
  initialContent: string;
  onChange: (content: string) => void;
  readOnly?: boolean;
}

const ConfigEditor: React.FC<ConfigEditorProps> = ({ initialContent, onChange, readOnly }) => {
  return (
    <textarea
      defaultValue={initialContent}
      readOnly={!!readOnly}
      onChange={e => onChange(e.target.value)}
      rows={12}
      style={{ width: '100%', fontFamily: 'monospace' }}
    />
  );
};

export default ConfigEditor;

